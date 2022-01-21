


'''Notes:

- Try use pep8
- pip install black.  within code:  black .

'''



mylist = [1,2,3,4,5]

mylist[:10] = [1,2,3,4,5]




def train_model(dataset, epochs=1, max_steps=10000,
                lr=0.001, log_steps=1000):
    # dataset is an array of sinusoid_generator functions
    model = SineModel()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    total_steps = 0
    for epoch in range(epochs):
        losses = []
        total_loss = 0
        start = time.time()
        for i, sinusoid_generator in enumerate(dataset):
            total_steps += 1
            if total_steps > max_steps:
                break
            x, y = sinusoid_generator.batch()
            loss = train_batch(x, y, model, optimizer)
            total_loss += loss
            curr_loss = total_loss / (i + 1.0)
            losses.append(curr_loss)

            if i % log_steps == 0 and i > 0:
                print('Step {}: loss = {}, Time to run {} steps = {:.2f} seconds'.format(
                    i, curr_loss, log_steps, time.time() - start))
                start = time.time()
        plt.plot(losses)
        plt.title('Loss Vs Time steps')
        plt.show()
    return model


# --- MAML ---
def train_maml(model, epochs, dataset, lr_inner=0.01, batch_size=1, log_steps=1000):
    # There seems to be no parameter for β. I guess we just scale the gradients
    '''Train using the MAML setup.
    The comments in this function that start with:
        Step X:
    Refer to a step described in the Algorithm 1 of the paper.
    Args:
        model: A model.
        epochs: Number of epochs used for training.
        dataset: A dataset used for training.
        lr_inner: Inner learning rate (alpha in Algorithm 1). Default value is 0.01.
        batch_size: Batch size. Default value is 1. The paper does not specify
            which value they use.
        log_steps: At every `log_steps` a log message is printed.
    Returns:
        A strong, fully-developed and trained maml.
    '''
    optimizer = keras.optimizers.Adam()

    # Step 2: instead of checking for convergence, we train for a number
    # of epochs
    for _ in range(epochs):
        total_loss = 0
        losses = []
        start = time.time()
        # Step 3 and 4
        for i, t in enumerate(random.sample(dataset, len(dataset))):
            x, y = np_to_tensor(t.batch())
            model.forward(x)  # run forward pass to initialize weights
            with tf.GradientTape() as test_tape:
                # test_tape.watch(model.trainable_variables)
                # Step 5
                sum_test_loss = 0.

                for task in batch:
                    model_copy = copy_model(model, x)
                    for inner_step in range(num_inner_steps):

                        with tf.GradientTape() as train_tape:
                            train_loss, _ = compute_loss(model_copy, x, y)
                            # Step 6. Use gradients to calculate the next kernel and bias
                            gradients = train_tape.gradient(train_loss, model_copy.trainable_variables)
                            # gradiants is an array of gradient for each kernal and bias. You can see its
                            # constitution in for loop.
                            k = 0
                            model_copy = copy_model(model_copy, x)
                            for j in range(len(model_copy.layers)):
                                model_copy.layers[j].kernel = tf.subtract(model_copy.layers[j].kernel,
                                                                          tf.multiply(lr_inner, gradients[k]))
                                model_copy.layers[j].bias = tf.subtract(model_copy.layers[j].bias,
                                                                        tf.multiply(lr_inner, gradients[k + 1]))
                                k += 2
                            # Step 8  Q: Here it really seems like it's applying one θi's loss to the model, instead of
                            # a sum!
                    test_loss, logits = compute_loss(model_copy, x, y)
                    sum_test_loss += test_loss
                # Step 8.    test_loss here is a number; gradients is an array of [1,n] tensors
                gradients = test_tape.gradient(sum_test_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Logs
            total_loss += test_loss
            loss = total_loss / (i + 1.0)
            losses.append(loss)

            if i % log_steps == 0 and i > 0:
                print('Step {}: loss = {}, Time to run {} steps = {}'.format(i, loss, log_steps, time.time() - start))
                start = time.time()
        plt.plot(losses)
        plt.show()
