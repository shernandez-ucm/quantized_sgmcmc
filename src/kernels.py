import jax
import jax.numpy as jnp


def sgld(key, params,grads, dt):
    gamma,eps=0.9,1e-6
    key,subkey=jax.random.split(key)
    noise=jax.tree_util.tree_map(lambda p: jax.random.normal(key=subkey,shape=p.shape), params)
    params=jax.tree_util.tree_map(lambda p, g,n: p-dt*g+jnp.sqrt(2*dt)*n, params, grads,noise)
    return key, params

def sgld_momemtum(key, params, momemtum,grads, dt):
    gamma,eps=0.9,1e-6
    key,subkey=jax.random.split(key)
    momemtum=jax.tree_util.tree_map(lambda m,g : gamma*m+(1.-gamma)*g,momemtum,grads)
    noise=jax.tree_util.tree_map(lambda p: jax.random.normal(key=subkey,shape=p.shape), params)
    params=jax.tree_util.tree_map(lambda p, g,m,n: p-dt*m+jnp.sqrt(2*dt)*n, params, grads,momemtum,noise)
    return key, params,momemtum

def sgd_momemtum(key, params, momemtum,grads, dt):
    gamma,eps=0.9,1e-6
    key,subkey=jax.random.split(key)
    momemtum=jax.tree_util.tree_map(lambda m,g : gamma*m+(1.-gamma)*g,momemtum,grads)
    params=jax.tree_util.tree_map(lambda p, m: p-dt*m, params,momemtum)
    return key, params,momemtum

def psgld_momemtum(key, params, momemtum,grads, dt):
    gamma,eps=0.9,1e-6
    key,subkey=jax.random.split(key)
    squared_grads=jax.tree_util.tree_map(lambda g: jnp.square(g),grads)
    momemtum=jax.tree_util.tree_map(lambda m,s : gamma*m+(1-gamma)*s,momemtum,squared_grads)
    noise=jax.tree_util.tree_map(lambda p: jax.random.normal(key=subkey,shape=p.shape), params)
    params=jax.tree_util.tree_map(lambda p, g,m,n: p-0.5*dt*g/(m+eps)+jnp.sqrt(dt)*n, params, grads,momemtum,noise)
    return key, params,momemtum