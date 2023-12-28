import jax
import jax.numpy as jnp

hyperparams = [2.0, 0.25]

def jax_stable_exp(z,s=1,axis=0):
    z = s*z
    z = z- z.max(axis)
    z = jnp.exp(z)
    return z

def mrp_to_rot(vec):
    vec_mag = vec @ vec
    vec_mag_num = (1-vec_mag)
    vec_mag_den = ((1+vec_mag)**2)
    x,y,z = vec
    K = jnp.array(
           [[  0, -z,  y ],
            [  z,  0, -x ],
            [ -y,  x,  0 ]])
    R1 = jnp.eye(3) - ( ((4*vec_mag_num)/vec_mag_den) * K) + ((8/vec_mag_den) * (K @ K))
    R2 = jnp.eye(3)

    Rest = jnp.where(vec_mag > 1e-12,R1,R2)
    return Rest

def axangle_to_rot(axangl):
    scale = jnp.sqrt(axangl @ axangl)
    vec = axangl/scale
    x,y,z = vec
    K = jnp.array(
           [[  0, -z,  y ],
            [  z,  0, -x ],
            [ -y,  x,  0 ]])
    ctheta = jnp.cos(scale)
    stheta = jnp.sin(scale)
    R1 = jnp.eye(3) + stheta*K + (1-ctheta)*(K @ K)
    R2 = jnp.eye(3)
    Rest = jnp.where(scale > 1e-12,R1.T, R2)
    return Rest

def quat_to_rot(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z

    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    R1 = jnp.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
    R2 = jnp.eye(3)
    return jnp.where(Nq > 1e-12,R1,R2)

# core rendering function
def render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3):
    # precision is fully parameterized by triangle matrix
    # we use upper triangle for compatibilize with sklearn
    prec = jnp.triu(prec_full)

    # if doing normalized weights for proper GMM
    # typically not used for shape reconstruction
    #weights = jnp.exp(weights_log)
    #weights = weights/weights.sum()

    # gets run per gaussian with [precision, log(weight), mean]
    def perf_idx(prcI,w,meansI):
        # math is easier with lower triangle
        prc = prcI.T

        # gaussian scale
        # could be useful for log likelihood but not used here
        div = jnp.prod(jnp.diag(jnp.abs(prc))) + 1e-20
        
        # gets run per ray
        def perf_ray(r_t):
            # unpack the ray (r) and position (t)
            r = r_t[0]
            t = r_t[1]
            
            # shift the mean to be relative to ray start
            p =  meansI - t

            # compute \sigma^{-0.5} p, which is reused
            projp = prc @ p

            # compute v^T \sigma^{-1} v
            vsv = ((prc @ r)**2).sum()

            # compute p^T \sigma^{-1} v
            psv = ((projp) * (prc@r)).sum()

            # compute the surface normal as \sigma^{-1} p
            projp2 = prc.T @ projp

            # distance to get maximum likelihood point for this gaussian
            # scale here is based on r! 
            # if r = [x, y, 1], then depth. if ||r|| = 1, then distance
            res = (psv)/(vsv)
            
            # get the intersection point
            v = r * res - p

            # compute intersection's unnormalized Gaussian log likelihood
            d0 = ((prc @ v)**2).sum()# + 3*jnp.log(jnp.pi*2)
            
            # multiply by the weight
            d2 = -0.5*d0 + w
            
            # if you wanted real probability
            #d3 =  d2 + jnp.log(div) #+ 3*jnp.log(res)

            # compute a normalized normal
            norm_est = projp2/jnp.linalg.norm(projp2)
            norm_est = jnp.where(r@norm_est < 0,norm_est,-norm_est)

            # return ray distance, gaussian distance, normal
            return res, d2, norm_est
        
        # runs parallel for each ray across each gaussian
        res,d2,projp  = jax.vmap((perf_ray))(camera_starts_rays)

        return res, d2,projp
    
    # runs parallel for gaussian
    zs,stds,projp = jax.vmap(perf_idx)(prec,weights_log,means) 

    # alpha is based on distance from all gaussians
    est_alpha = 1-jnp.exp(-jnp.exp(stds).sum(0) )

    # points behind camera should be zero
    # BUG: est_alpha should also use this
    sig1 = (zs > 0)# sigmoid
    
    # compute the algrebraic weights in the paper
    w = sig1*jnp.nan_to_num(jax_stable_exp(-zs*beta_2 + beta_3*stds))+1e-20

    # normalize weights
    wgt  = w.sum(0)
    div = jnp.where(wgt==0,1,wgt)
    w = w/div

    # compute weighted z and normal
    init_t=  (w*jnp.nan_to_num(zs)).sum(0)
    est_norm = (projp * w[:,:,None]).sum(axis=0)
    est_norm = est_norm/jnp.linalg.norm(est_norm,axis=1,keepdims=True)

    # return z, alpha, normal, and the weights
    # weights can be used to compute color, DINO features, or any other per-Gaussian property
    return init_t,est_alpha,est_norm,w


# this version has no hyperparameters
# and does classic alpha compositing down z-order
# it's typically 3x slower than the above
# and seems to behave slightly worse
# but serves as a direct substitute.
# I will document this with experiments
# in a preprint sometime in June/July '23
def render_func_rays_hffm(means, prec_full, weights_log, camera_starts_rays):
    prec = jnp.triu(prec_full)
    #weights = jnp.exp(weights_log)
    #weights = weights/weights.sum()

    def perf_idx(prcI,w,meansI):
        prc = prcI.T
        #prc = jnp.diag(jnp.sign(jnp.diag(prc))) @ prc
        div = jnp.prod(jnp.diag(jnp.abs(prc))) + 1e-20

        def perf_ray(r_t):
            r = r_t[0]
            t = r_t[1]
            p =  meansI -t 

            projp = prc @ p
            vsv = ((prc @ r)**2).sum()
            psv = ((projp) * (prc@r)).sum()
            projp2 = prc.T @ projp
            # linear
            res = (psv)/(vsv)
            
            v = r * res - p

            d0 = ((prc @ v)**2).sum()# + 3*jnp.log(jnp.pi*2)
            d2 = -0.5*d0 + w
            #d3 =  d2 + jnp.log(div) #+ 3*jnp.log(res)
            norm_est = projp2/jnp.linalg.norm(projp2)
            norm_est = jnp.where(r@norm_est < 0,norm_est,-norm_est)
            return res,d2,norm_est
        res,d2,projp  = jax.vmap((perf_ray))(camera_starts_rays) # jit perf
        return res, d2,projp

    zs,stds,projp = jax.vmap(perf_idx)(prec,weights_log,means)  # jit perf

    # compositing
    sample_density = jnp.exp(stds)  # simplier but splottier
    def sort_w(z,densities):
        idxs = jnp.argsort(z,axis=0)
        order_density = densities[idxs]
        order_summed_density = jnp.cumsum(order_density)
        order_prior_density =  order_summed_density - order_density
        ea = 1 - jnp.exp(-order_summed_density[-1])
        prior_density = jnp.zeros_like(densities)
        prior_density = prior_density.at[idxs].set(order_prior_density)
        transmit = jnp.exp(-prior_density)
        wout = transmit * (1-jnp.exp(-densities))
        return wout,ea
    w,est_alpha= jax.vmap(sort_w)(zs.T,sample_density.T)
    w = w.T

    wgt  = w.sum(0)
    div = jnp.where(wgt==0,1,wgt)
    w_n = w/div

    init_t=  (w_n*jnp.nan_to_num(zs)).sum(0)
    est_norm = (projp * w_n[:,:,None]).sum(axis=0)
    est_norm = est_norm/jnp.linalg.norm(est_norm,axis=1,keepdims=True)

    return init_t,est_alpha,est_norm,w
    
# axis angle rotations n * theta
def render_func_axangle(means, prec_full, weights_log, camera_rays, axangl, t, beta_2, beta_3):
    Rest = axangle_to_rot(axangl)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3)

# modified rod. parameters n * tan(theta/4)
def render_func_mrp(means, prec_full, weights_log, camera_rays, mrp, t, beta_2, beta_3):
    Rest = mrp_to_rot(mrp)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3)

# quaternions [cos(theta/2), sin(theta/2) * n]
def render_func_quat(means, prec_full, weights_log, camera_rays, quat, t, beta_2, beta_3):
    Rest = quat_to_rot(quat)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3)