import matplotlib.pyplot as plt

def plot_system(X, U):
    for i in range(3):
        plt.plot(X[:, 2*i], X[:, 2*i+1])
    plt.xlim(-2.1, 2.1)
    plt.ylim(-2.1, 2.1)

    plt.figure()
    plt.plot(U)
    plt.show()