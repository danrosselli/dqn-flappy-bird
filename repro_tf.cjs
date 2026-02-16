
const tf = require('@tensorflow/tfjs');

async function run() {
    await tf.setBackend('cpu'); // Or webgl, but cpu is easier for node
    console.log('Backend:', tf.getBackend());

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, inputShape: [5], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 2, activation: 'linear' }));

    const model2 = tf.sequential();
    model2.add(tf.layers.dense({ units: 10, inputShape: [5], activation: 'relu' }));
    model2.add(tf.layers.dense({ units: 2, activation: 'linear' }));

    // Simulate syncFrom logic
    console.log('Syncing models...');
    const sourceWeights = model.getWeights(); // [t1, t2, ...]

    // Check if they are unique tensors
    console.log('Source weights[0] ID:', sourceWeights[0].id);

    const clonedWeights = sourceWeights.map(t => t.clone());
    console.log('Cloned weights[0] ID:', clonedWeights[0].id);

    model2.setWeights(clonedWeights);

    // Dispose source weights (as done in NestedModel.js syncFrom)
    console.log('NOT Disposing source weights (Fix applied)...');
    // sourceWeights.forEach(t => t.dispose()); <--- REMOVED

    // Check if model (source) is still alive
    console.log('Checking source model...');
    try {
        model.predict(tf.zeros([1, 5])).dispose();
        console.log('Source model OK');
    } catch (e) {
        console.error('Source model BROKEN:', e.message);
    }

    // Dispose cloned weights (as done in NestedModel.js syncFrom indirectly? or NOT done?)
    // In strict sense, NestedModel.js does:
    // sourceWeights.forEach(t => t.dispose());
    // clonedWeights.forEach(t => t.dispose());

    console.log('Disposing cloned weights...');
    clonedWeights.forEach(t => t.dispose());

    // Check if model2 (target) is still alive
    console.log('Checking target model...');
    try {
        model2.predict(tf.zeros([1, 5])).dispose();
        console.log('Target model OK');
    } catch (e) {
        console.error('Target model BROKEN:', e.message);
    }

    // --- Check Optimizer Issue ---
    console.log('\nChecking Optimizer logic...');
    const layer = model.layers[0];
    const vars = layer.getWeights(); // This returns Tensors/Values
    console.log('Layer getWeights() returns:', vars.map(v => v.constructor.name));

    // vs Trainable Weights
    const trainable = layer.trainableWeights;
    // In tfjs-layers, accessing .trainableWeights might return LayerVariable objects, not Tensors?
    // Let's print what we get.
    console.log('Layer trainableWeights length:', trainable ? trainable.length : 'undefined');
    if (trainable && trainable.length > 0) {
        console.log('Item 0 type:', trainable[0].constructor.name);
        // LayerVariable usually has .read() or .val
    }

    vars.forEach(t => t.dispose()); // clean up snapshots
}

run();
