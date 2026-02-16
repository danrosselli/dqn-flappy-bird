
const tf = require('@tensorflow/tfjs');

async function checkLayerVariable() {
    const layer = tf.layers.dense({ units: 1, inputShape: [1] });
    // Build layer to initialize weights
    layer.apply(tf.zeros([1, 1]));

    console.log('Trainable weights length:', layer.trainableWeights.length);
    const weight = layer.trainableWeights[0];
    console.log('Weight constructor:', weight.constructor.name);
    console.log('Keys:', Object.keys(weight));

    // Check for underlying variable
    if (weight.val) {
        console.log('Has .val property (Variable?):', weight.val instanceof tf.Variable || (weight.val.constructor && weight.val.constructor.name === 'Variable'));
    }

    // Check if we can pass it to minimize
    const opt = tf.train.sgd(0.1);
    const x = tf.variable(tf.tensor([1]));

    try {
        opt.minimize(() => x.square(), false, [x]);
        console.log('Minimize with tf.Variable works works.');
    } catch (e) { console.error('Minimize tf.Variable failed:', e.message); }

    try {
        // try passing LayerVariable directly
        // usually LayerVariable wraps a tensor/variable.
        // It seems LayerVariable has a `read()` method.
        // But for minimize, we need the variable itself.
        // Let's see if we can get it.
    } catch (e) { }
}

checkLayerVariable();
