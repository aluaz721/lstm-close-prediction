"""Train/evaluate loops, ported from the original notebook's train_model /
test_model / predict functions and generalized into a reusable `fit`."""

import torch


def train_epoch(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


def test_epoch(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    return total_loss / num_batches


def predict(data_loader, model):
    """Run inference over a data loader, preserving row order (the loader
    must not shuffle)."""
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output


def fit(train_loader, test_loader, model, loss_function, optimizer, num_epochs, verbose=True):
    """Train for `num_epochs`, evaluating on `test_loader` after every epoch.

    Returns a history dict with the pre-training test loss included as
    epoch 0, matching what the original notebook printed and plotted.
    """
    history = {"train_loss": [], "test_loss": []}

    initial_test_loss = test_epoch(test_loader, model, loss_function)
    history["test_loss"].append(initial_test_loss)
    if verbose:
        print(f"Untrained test loss: {initial_test_loss}")

    for epoch in range(num_epochs):
        train_loss = train_epoch(train_loader, model, loss_function, optimizer)
        test_loss = test_epoch(test_loader, model, loss_function)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        if verbose:
            print(f"Epoch {epoch}: train_loss={train_loss} test_loss={test_loss}")

    return history
