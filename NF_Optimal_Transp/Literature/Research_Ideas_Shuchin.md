## Research Ideas


### deep image prior
deep image network act as a regularize, does not require an external regularizer, instead of using a deterministic dictionary such as curvelets, 
we use CNN as a dictionary and estimate dictionary for fix coefficients z. A couple of direction we can take: 

#### Basic work
what about if we learn both the dictionary and z as we do in seismic in real dictionary learning. Any benefit of 
doing that? we might define more data specific z instead of gaussian. Also is there any benefits in transfer learning if we do this 
over theta versus theta/z.  Is there a way we can create dictinary like curvelet which can be generalized but using networks and what are the benefits?

#### Advance work
how about we define this G(z) in a way that this is invertible like we do with normalizing flow. In that case, once we do dictionary 
learning and coefficient minimization, we can use an invertible network on any new x to get a map estimate. If we only do theta then invertible network gives gaussian 
noise, but if we minimize over z and theta then output if going to be the solved problem over which we are interested in. This way we can use this network as cheap transfer 
learning approach or directly apply it to get new images. Alternate minimization is needed.


#### Invert to learn to invert
scalable to large dimension instead of 2D, using reversible network with invertible theme, dont need to same intermediate steps
 
