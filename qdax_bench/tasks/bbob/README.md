# Black-Box Optimization Benchmark

Implementation of the BBOB benchmark functions applied to QD, based on [Maxence Faldor](https://github.com/maxencefaldor)'s implementations in [learned-qd](https://github.com/maxencefaldor/learned-qd) and [evosax](https://github.com/RobertTLange/evosax).

## Descriptor Range Estimation

Given a random projection defined by:

```python
def gaussian_random_projection(self, key: jax.Array) -> jax.Array:
    return jax.random.normal(
        key,
        shape=(self.descriptor_size, self.num_dims),
    ) / jnp.sqrt(self.descriptor_size)
```

We construct the projection matrix:

```python
descriptor_params = self.gaussian_random_projection(key_desc)
```

and apply it to input vectors `x`:

```python
descriptor = jax.vmap(lambda x_i: descriptor_params @ x_i)(x)
```

### Range Analysis

Assume:
- Each input vector $x_i \in \mathbb{R}^{\text{num\_dims}}$ has components in the range $[a, b]$.
- The projection matrix `descriptor_params` has entries sampled i.i.d. from $\mathcal{N}\left(0, \frac{1}{\text{descriptor\_size}}\right)$.

Each output descriptor component is a linear combination of the form:
$$
d_i = \sum_{j=1}^{\text{num\_dims}} a_{ij} \cdot x_j, \quad \text{where } a_{ij} \sim \mathcal{N}\left(0, \frac{1}{\text{descriptor\_size}}\right)
$$

This results in:
- Mean $\mathbb{E}[d_i] = 0$
- Variance:
$$
\mathrm{Var}[d_i] = \sum_{j=1}^{\text{num\_dims}} \frac{1}{\text{descriptor\_size}} x_j^2 = \frac{1}{\text{descriptor\_size}} \|x_i\|^2
$$

Since $x_j \in [a, b]$, the worst-case squared norm is:
$$
\|x_i\|^2 \leq \text{num\_dims} \cdot \max(a^2, b^2)
$$

Therefore, the standard deviation of each descriptor component is bounded by:
$$
\sigma \leq \sqrt{ \frac{\text{num\_dims} \cdot \max(a^2, b^2)}{\text{descriptor\_size}} }
$$

Assuming a Gaussian distribution, approximately 95% of values lie within $2\sigma$, so the **probable range** of each descriptor component is:
$$
\left[
-2 \cdot \sqrt{ \frac{\text{num\_dims} \cdot \max(a^2, b^2)}{\text{descriptor\_size}} },\ 
2 \cdot \sqrt{ \frac{\text{num\_dims} \cdot \max(a^2, b^2)}{\text{descriptor\_size}} }
\right]
$$

Or more conservatively (99% of mass), within $3\sigma$:
$$
\left[
-3 \cdot \sqrt{ \frac{\text{num\_dims} \cdot \max(a^2, b^2)}{\text{descriptor\_size}} },\ 
3 \cdot \sqrt{ \frac{\text{num\_dims} \cdot \max(a^2, b^2)}{\text{descriptor\_size}} }
\right]
$$

### Example

If:
- `x_range = [-5, 5]`, so $\max(a^2, b^2) = 25$
- `num_dims = 20`
- `descriptor_size = 5`

Then:
$$
\sigma \leq \sqrt{ \frac{20 \cdot 25}{5} } = \sqrt{100} = 10 \quad \Rightarrow \quad \text{Probable range } \approx [-20, 20]
$$
