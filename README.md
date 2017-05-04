# room-shape
MLP learns (boxy) room shape given mode frequencies, or vice versa


Just sharing a new "toy" result, that relates to function spaces, acoustics, and machine learning:

During [Karlheinz Brandenburg's visit](http://www.belmont.edu/burs/), he remarked that learning room shapes from the sound of the room is still an open question. So yesterday, "for fun" I decided to try the easiest possible thing I could think of, the most "ideal" if you will:  wrote a NN system that learns to use a set of room mode frequencies to predict (boxy) room dimensions, or vice versa (shown in pic, red=target, green=predicted), within 5%.   It even learns to sort the freqs in ascending order.  i.e. it learns the "Rayleigh Equation" for 3D standing waves... 

<a href="https://www.codecogs.com/eqnedit.php?latex=f_{nx,ny,nz}&space;=&space;{v_s&space;\over&space;2}&space;\sqrt{&space;\left(&space;{n_x&space;\over&space;L}&space;\right)^2&space;&plus;&space;\left(&space;{n_y&space;\over&space;W}&space;\right)^2&space;&plus;&space;\left(&space;{n_z&space;\over&space;H}&space;\right)^2&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_{nx,ny,nz}&space;=&space;{v_s&space;\over&space;2}&space;\sqrt{&space;\left(&space;{n_x&space;\over&space;L}&space;\right)^2&space;&plus;&space;\left(&space;{n_y&space;\over&space;W}&space;\right)^2&space;&plus;&space;\left(&space;{n_z&space;\over&space;H}&space;\right)^2&space;}" title="f_{nx,ny,nz} = {v_s \over 2} \sqrt{ \left( {n_x \over L} \right)^2 + \left( {n_y \over W} \right)^2 + \left( {n_z \over H} \right)^2 }" /></a>

...both 'forwards' or 'backwards'. 

Interestingly, trying to 'help' the network by using squared frequencies, inverse-(squared)-dimensions, etc. actually gave worse performance than letting the network learn the nonlinearities itself. (???)

Sample room shown in the picture, used a training set of 200,000 random rooms.

Of what possible practical utility is this? Not really sure. Although, it does a fairly decent job learning the room shape even using a smaller random subsample of the available mode frequencies. :-)  If anything, it highlights a weakness of data-driven modeling: there's no way you'll measure 200,000 rooms in real life!

Taking this further, would require
Analyzing the weights of the network in detail, to see how it represents the function approximation.  Which would be worth doing!
