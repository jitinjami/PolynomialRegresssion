import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np

def create_dataset(w_star, x_range, sample_size, sigma, seed=None):
    random_state = np.random.RandomState(seed)
    x = random_state.uniform(x_range[0], x_range[1],(sample_size)) 
    X = np.zeros((sample_size , w_star.shape[0]))
    for i in range(sample_size): 
        X[i, 0] = 1.
        for j in range(1, w_star.shape[0]): 
            X[i, j] = x[i]**j
    y = X.dot(w_star)
    if sigma > 0:
        y += random_state.normal(0.0 , sigma , sample_size )
    return X, x, y

x_range = [-3,2]
w_star = np.array([-8,-4,2,1])
w_star = w_star.T
sigma = 0.5
seed_train = 0
seed_validation = 1
sample_size_train = 100
sample_size_validation = 100

X_train, x_plot, y_train = create_dataset(w_star, x_range, sample_size_train, sigma, seed_train)
X_validate, x_plot1, y_validate = create_dataset(w_star, x_range, sample_size_validation, sigma, seed_validation)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("y", fontsize=16)
ax.set_title(f"Training and Validation Set")
ax.plot(x_plot, y_train, ".")
ax.plot(x_plot1, y_validate, ".")
ax.legend(['Training set','Validation set'])
plt.savefig(f'train_validation_set.png')
plt.close()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.Linear(4,1)
model = model.to(DEVICE)
loss_function = nn.MSELoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
X_train_tensor = torch.from_numpy(X_train.reshape(sample_size_train,4)).float().to(DEVICE)
y_train_tensor = torch.from_numpy(y_train.reshape(sample_size_train,1)).float().to(DEVICE)
X_validate_tensor = torch.from_numpy(X_validate.reshape(sample_size_validation,4)).float().to(DEVICE)
y_validate_tensor = torch.from_numpy(y_validate.reshape(sample_size_validation,1)).float().to(DEVICE)

num_steps = 1500
training_losses = []
validation_losses = []

for step in range(num_steps):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_train_tensor)
    loss = loss_function(y_pred, y_train_tensor)
    training_losses.append(loss)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_validate_tensor)
        val_loss = loss_function(y_val_pred, y_validate_tensor)
        validation_losses.append(val_loss)

iterations = range(num_steps)
fig, ax = plt.subplots()
ax.set_xlabel("Iterations", fontsize=16)
ax.set_ylabel("Losses", fontsize=16)
ax.set_title(f"Training and validation losses")
ax.plot(iterations, training_losses, ".")
ax.plot(iterations, validation_losses, ".")
ax.legend(['Training loss','Validation loss'])
plt.savefig(f'losses.png')
plt.close()

w_star_data = []
w_hat_data = []
x_values = np.arange(-3,2,0.01)

for value in x_values:
        x1_values = np.zeros( w_star.shape[0])
        for i in range(len(x1_values)):
                x1_values[i] = value**i
        w_star_values = np.dot(x1_values,w_star)
        w_star_data.append(w_star_values)

        w_hat_values = np.dot(x1_values, model.weight.detach().numpy().T) + model.bias.detach().numpy()[0]
        w_hat_data.append(w_hat_values)

fig, ax = plt.subplots()
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("y", fontsize=16)
ax.plot(x_values, w_star_data, ".")
ax.plot(x_values, w_hat_data, ".")
ax.set_title(f"W_star and W_hat")
ax.legend(['W_star','W_hat'])
plt.savefig(f'w_star_w_hat.png')
plt.close()

#Changing sampling sizes
sampling_sizes = [5,10,50,100]
for sampling_size in sampling_sizes:
    x_range = [-3,2]
    w_star = np.array([-8,-4,2,1])
    w_star = w_star.T
    sigma = 0.5
    seed_train = 0
    seed_validation = 1
    sample_size_train = sampling_size
    sample_size_validation = 100

    X_train, x_plot, y_train = create_dataset(w_star, x_range, sample_size_train, sigma, seed_train)
    X_validate, x_plot1, y_validate = create_dataset(w_star, x_range, sample_size_validation, sigma, seed_validation)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.set_title(f"Training and Validation Set with sampling_size = {sample_size_train}")
    ax.plot(x_plot, y_train, ".")
    ax.plot(x_plot1, y_validate, ".")
    ax.legend(['Training set','Validation set'])
    plt.savefig(f'train_validation_set_{sample_size_train}.png')
    plt.close()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(4,1)
    model = model.to(DEVICE)
    loss_function = nn.MSELoss()
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    X_train_tensor = torch.from_numpy(X_train.reshape(sample_size_train,4)).float().to(DEVICE)
    y_train_tensor = torch.from_numpy(y_train.reshape(sample_size_train,1)).float().to(DEVICE)
    X_validate_tensor = torch.from_numpy(X_validate.reshape(sample_size_validation,4)).float().to(DEVICE)
    y_validate_tensor = torch.from_numpy(y_validate.reshape(sample_size_validation,1)).float().to(DEVICE)

    num_steps = 1500
    training_losses = []
    validation_losses = []

    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train_tensor)
        loss = loss_function(y_pred, y_train_tensor)
        training_losses.append(loss)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_validate_tensor)
            val_loss = loss_function(y_val_pred, y_validate_tensor)
            validation_losses.append(val_loss)

    iterations = range(num_steps)
    fig, ax = plt.subplots()
    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel("Losses", fontsize=16)
    ax.set_title(f"Training and validation losses with sampling_size = {sample_size_train}")
    ax.plot(iterations, training_losses, ".")
    ax.plot(iterations, validation_losses, ".")
    ax.legend(['Training loss','Validation loss'])
    plt.savefig(f'losses_{sample_size_train}.png')
    plt.close()

    w_star_data = []
    w_hat_data = []
    x_values = np.arange(-3,2,0.01)

    for value in x_values:
            x1_values = np.zeros( w_star.shape[0])
            for i in range(len(x1_values)):
                    x1_values[i] = value**i
            w_star_values = np.dot(x1_values,w_star)
            w_star_data.append(w_star_values)

            w_hat_values = np.dot(x1_values, model.weight.detach().numpy().T) + model.bias.detach().numpy()[0]
            w_hat_data.append(w_hat_values)

    fig, ax = plt.subplots()
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.plot(x_values, w_star_data, ".")
    ax.plot(x_values, w_hat_data, ".")
    ax.set_title(f"W_star and W_hat with sampling_size = {sample_size_train}")
    ax.legend(['W_star','W_hat'])
    plt.savefig(f'w_star_w_hat_{sample_size_train}.png')
    plt.close()


#Changing sigma
sigmas = [0.5,1,2,4,8]
for sigma1 in sigmas:
    x_range = [-3,2]
    w_star = np.array([-8,-4,2,1])
    w_star = w_star.T
    sigma = sigma1
    seed_train = 0
    seed_validation = 1
    sample_size_train = 100
    sample_size_validation = 100

    X_train, x_plot, y_train = create_dataset(w_star, x_range, sample_size_train, sigma, seed_train)
    X_validate, x_plot1, y_validate = create_dataset(w_star, x_range, sample_size_validation, sigma, seed_validation)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.set_title(f"Training and Validation Set with sigma = {sigma}")
    ax.plot(x_plot, y_train, ".")
    ax.plot(x_plot1, y_validate, ".")
    ax.legend(['Training set','Validation set'])
    plt.savefig(f'train_validation_set_{sigma}.png')
    plt.close()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(4,1)
    model = model.to(DEVICE)
    loss_function = nn.MSELoss()
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    X_train_tensor = torch.from_numpy(X_train.reshape(sample_size_train,4)).float().to(DEVICE)
    y_train_tensor = torch.from_numpy(y_train.reshape(sample_size_train,1)).float().to(DEVICE)
    X_validate_tensor = torch.from_numpy(X_validate.reshape(sample_size_validation,4)).float().to(DEVICE)
    y_validate_tensor = torch.from_numpy(y_validate.reshape(sample_size_validation,1)).float().to(DEVICE)

    num_steps = 1500
    training_losses = []
    validation_losses = []

    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train_tensor)
        loss = loss_function(y_pred, y_train_tensor)
        training_losses.append(loss)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_validate_tensor)
            val_loss = loss_function(y_val_pred, y_validate_tensor)
            validation_losses.append(val_loss)

    iterations = range(num_steps)
    fig, ax = plt.subplots()
    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel("Losses", fontsize=16)
    ax.set_title(f"Training and validation losses with sigma = {sigma}")
    ax.plot(iterations, training_losses, ".")
    ax.plot(iterations, validation_losses, ".")
    ax.legend(['Training loss','Validation loss'])
    plt.savefig(f'losses_{sigma}.png')
    plt.close()

    w_star_data = []
    w_hat_data = []
    x_values = np.arange(-3,2,0.01)

    for value in x_values:
            x1_values = np.zeros( w_star.shape[0])
            for i in range(len(x1_values)):
                    x1_values[i] = value**i
            w_star_values = np.dot(x1_values,w_star)
            w_star_data.append(w_star_values)

            w_hat_values = np.dot(x1_values, model.weight.detach().numpy().T) + model.bias.detach().numpy()[0]
            w_hat_data.append(w_hat_values)

    fig, ax = plt.subplots()
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.plot(x_values, w_star_data, ".")
    ax.plot(x_values, w_hat_data, ".")
    ax.set_title(f"W_star and W_hat with sigma = {sigma}")
    ax.legend(['W_star','W_hat'])
    plt.savefig(f'w_star_w_hat_{sigma}.png')
    plt.close()

#Bonus
x_range = [-3,2]
w_star = np.array([-8,-4,2,1,2])
w_star = w_star.T
sigma = 0.5
seed_train = 0
seed_validation = 1
sample_size_train = 10
sample_size_validation = 100

X_train, x_plot, y_train = create_dataset(w_star, x_range, sample_size_train, sigma, seed_train)
X_validate, x_plot1, y_validate = create_dataset(w_star, x_range, sample_size_validation, sigma, seed_validation)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("y", fontsize=16)
ax.set_title(f"Training and Validation Set")
ax.plot(x_plot, y_train, ".")
ax.plot(x_plot1, y_validate, ".")
ax.legend(['Training set','Validation set'])
plt.savefig(f'train_validation_set_4th.png')
plt.close()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.Linear(5,1)
model = model.to(DEVICE)
loss_function = nn.MSELoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
X_train_tensor = torch.from_numpy(X_train.reshape(sample_size_train,5)).float().to(DEVICE)
y_train_tensor = torch.from_numpy(y_train.reshape(sample_size_train,1)).float().to(DEVICE)
X_validate_tensor = torch.from_numpy(X_validate.reshape(sample_size_validation,5)).float().to(DEVICE)
y_validate_tensor = torch.from_numpy(y_validate.reshape(sample_size_validation,1)).float().to(DEVICE)

num_steps = 1500
training_losses = []
validation_losses = []

for step in range(num_steps):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_train_tensor)
    loss = loss_function(y_pred, y_train_tensor)
    training_losses.append(loss)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_validate_tensor)
        val_loss = loss_function(y_val_pred, y_validate_tensor)
        validation_losses.append(val_loss)

iterations = range(num_steps)
fig, ax = plt.subplots()
ax.set_xlabel("Iterations", fontsize=16)
ax.set_ylabel("Losses", fontsize=16)
ax.set_title(f"Training and validation losses")
ax.plot(iterations, training_losses, ".")
ax.plot(iterations, validation_losses, ".")
ax.legend(['Training loss','Validation loss'])
plt.savefig(f'losses_4th.png')
plt.close()

w_star_data = []
w_hat_data = []
x_values = np.arange(-3,2,0.01)

for value in x_values:
        x1_values = np.zeros( w_star.shape[0])
        for i in range(len(x1_values)):
                x1_values[i] = value**i
        w_star_values = np.dot(x1_values,w_star)
        w_star_data.append(w_star_values)

        w_hat_values = np.dot(x1_values, model.weight.detach().numpy().T) + model.bias.detach().numpy()[0]
        w_hat_data.append(w_hat_values)

fig, ax = plt.subplots()
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("y", fontsize=16)
ax.plot(x_values, w_star_data, ".")
ax.plot(x_values, w_hat_data, ".")
ax.set_title(f"W_star and W_hat")
ax.legend(['W_star','W_hat'])
plt.savefig(f'w_star_w_hat_4th.png')
plt.close()