Idea:

Manifold is in high dim space
Neighbours are far away due to concentration of measure
Tangent estimation suffers from high bias

We enocde the manifold into a smaller space -- Latent dim
Then decode the manifold into the original space -- Ambient dim
If we encode the maniolfd into a latent dim == intrinsic dim, we get a flattening.
Flattening makes the manifold tangent space trivial; therefore the jacobian of the decoder is the tangent space of the manifold in ambient dim

Flattening destroys some information -- but that might not matter

Why do we want the decoder to be the same
    Cycle consistency enforces the preservation of the mutual information between input and translated images
    mappings are constrained to be orthonormal

    Mappings need to be continious -- this is true






Main points:

1. Why do we need the decoder to be the same?
   1. Because then we get invertibility on all of R^N -- can be used for sampling
2. Why is the basis trivial
3. Why do we get the angle of the tangent space
4. Regularising and things like that
We need a way to regularise: this paper is really good and helpful: 2305.01777v4


enoder decoder which share weights
Make sure that the encoder decoder are somehow injective?


so they have some kl_bound? We are talking about maps between spaces which have a different intrinsic dim


or we can regularise the loss -- but then we can easily just get the tangent of this manifold to analysis purposes better than LPCA


We simply Transpose the weights which guarantees orthonormality
We use leaky relu which is invertible -- leaky relu is not real-analytic -- so this needs some improvements

softplus -- is there a paper on this?


And the there is a way to invert the residual


ok, so now what about the tangent space reconstruction
