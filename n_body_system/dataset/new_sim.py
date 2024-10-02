import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(0)


class ChargedParticlesSim(object):
    def __init__(
        self,
        n_balls=5,
        box_size=5.0,
        loc_std=1.0,
        vel_norm=0.5,
        interaction_strength=1.0,
        noise_var=0.0,
        masses=None,
        force_field=None,
        G=1.0,  # Gravitational constant (can be scaled)
    ):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std * (float(n_balls) / 5.0) ** (1 / 3)
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var
        self.force_field = force_field
        self.G = G  # Gravitational constant

        if masses is None:
            self.masses = np.ones(n_balls)  # Default to unit mass if none provided
        else:
            self.masses = np.array(masses)
        self._charge_types = np.array([-1.0, 0.0, 1.0])
        self._delta_T = 0.01
        self._max_F = 0.1 / self._delta_T
        self.dim = 3

    def _l2(self, A, B):
        A_norm = (A**2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):
        with np.errstate(divide="ignore"):
            K = (
                0.5 * (self.masses * (vel**2).sum(axis=0)).sum()
            )  # Kinetic energy considering different masses
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r**2).sum())
                        U += 0.5 * self.interaction_strength * edges[i, j] / dist
                        U -= (
                            self.G * self.masses[i] * self.masses[j] / dist
                        )  # Gravitational potential energy
            return U + K

    def _clamp(self, loc, vel):
        assert np.all(loc < self.box_size * 3)
        assert np.all(loc > -self.box_size * 3)

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert np.all(loc <= self.box_size)
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        assert np.all(loc >= -self.box_size)
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(
        self, T=10000, sample_freq=10, charge_prob=[1.0 / 2, 0, 1.0 / 2]
    ):
        n = self.n_balls
        T_save = T
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0

        charges = np.random.choice(
            self._charge_types, size=(self.n_balls, 1), p=charge_prob
        )
        edges = charges.dot(charges.transpose())

        loc = np.zeros((T_save, self.dim, n))
        vel = np.zeros((T_save, self.dim, n))
        loc_next = np.random.randn(self.dim, n) * self.loc_std
        vel_next = np.random.randn(self.dim, n)
        v_norm = np.sqrt((vel_next**2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        with np.errstate(divide="ignore"):
            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3.0 / 2.0
            )

            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size, 0)
            F = (
                forces_size.reshape(1, n, n)
                * np.concatenate(
                    (
                        np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                            1, n, n
                        ),
                        np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                            1, n, n
                        ),
                        np.subtract.outer(loc_next[2, :], loc_next[2, :]).reshape(
                            1, n, n
                        ),
                    )
                )
            ).sum(axis=-1)

            # Gravitational forces
            F_gravity = np.zeros_like(F)
            for i in range(n):
                for j in range(i + 1, n):
                    r = loc_next[:, i] - loc_next[:, j]
                    dist = np.sqrt((r**2).sum())
                    force_mag = self.G * self.masses[i] * self.masses[j] / dist**2
                    force_dir = r / dist
                    F_gravity[:, i] -= force_mag * force_dir
                    F_gravity[:, j] += force_mag * force_dir

            F += F_gravity

            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            # Incorporate external force field
            if self.force_field is not None:
                external_forces = self.force_field(loc_next)
                F += external_forces / self.masses.reshape(1, -1)

            vel_next += self._delta_T * F / self.masses.reshape(1, -1)

            for i in range(1, T):
                loc_next += self._delta_T * vel_next

                loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()), 3.0 / 2.0
                )
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)

                F = (
                    forces_size.reshape(1, n, n)
                    * np.concatenate(
                        (
                            np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                                1, n, n
                            ),
                            np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                                1, n, n
                            ),
                            np.subtract.outer(loc_next[2, :], loc_next[2, :]).reshape(
                                1, n, n
                            ),
                        )
                    )
                ).sum(axis=-1)

                # Gravitational forces
                F_gravity = np.zeros_like(F)
                for i in range(n):
                    for j in range(i + 1, n):
                        r = loc_next[:, i] - loc_next[:, j]
                        dist = np.sqrt((r**2).sum())
                        force_mag = self.G * self.masses[i] * self.masses[j] / dist**2
                        force_dir = r / dist
                        F_gravity[:, i] -= force_mag * force_dir
                        F_gravity[:, j] += force_mag * force_dir

                F += F_gravity

                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F

                # Incorporate external force field
                if self.force_field is not None:
                    external_forces = self.force_field(loc_next)
                    F += external_forces / self.masses.reshape(1, -1)

                vel_next += self._delta_T * F / self.masses.reshape(1, -1)

            loc += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            return loc, vel, edges, charges


if __name__ == "__main__":

    # def complex_force_field(loc):
    #     F = np.zeros_like(loc)
    #     F[0, :] = -loc[1, :]
    #     F[1, :] = loc[0, :]
    #     F[2, :] = np.sin(loc[2, :])
    #     return F

    def spiraling_conservative_field(loc, k=1.0, m=0.1):
        F = np.zeros_like(loc)
        x = loc[0, :]
        y = loc[1, :]

        F[0, :] = k * x / (x**2 + y**2) - m * x
        F[1, :] = k * y / (x**2 + y**2) - m * y
        # No force in z-direction
        return F

    masses = [1.0, 2.0, 1.5, 0.5, 3.0]

    sim = ChargedParticlesSim(
        n_balls=5,
        loc_std=2,
        masses=masses,
        force_field=spiraling_conservative_field,
        G=1.0,
        box_size=20,
    )

    t = time.time()
    loc, vel, edges, charges = sim.sample_trajectory(T=10000, sample_freq=1)

    # print(edges)
    # print("Simulation time: {}".format(time.time() - t))
    # vel_norm = np.sqrt((vel**2).sum(axis=1))
    # plt.figure()
    # axes = plt.gca()
    # axes.set_xlim([-10.0, 10.0])
    # axes.set_ylim([-10.0, 10.0])
    # for i in range(loc.shape[-1]):
    #     plt.plot(loc[:, 0, i], loc[:, 1, i])
    #     plt.plot(loc[0, 0, i], loc[0, 1, i], "d")
    # plt.figure()
    # energies = [
    #     sim._energy(loc[i, :, :], vel[i, :, :], edges) for i in range(loc.shape[0])
    # ]
    # plt.plot(energies)
    # plt.show()
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # # Assuming loc, vel, edges, charges have been computed as before

    # # 3D Plot of Trajectories
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_xlim([-10.0, 10.0])
    # ax.set_ylim([-10.0, 10.0])
    # ax.set_zlim([-10.0, 10.0])

    # for i in range(loc.shape[-1]):
    #     ax.plot(loc[:, 0, i], loc[:, 1, i], loc[:, 2, i])
    #     ax.scatter(
    #         loc[0, 0, i],
    #         loc[0, 1, i],
    #         loc[0, 2, i],
    #         marker="o",
    #         label=f"Particle {i+1}",
    #     )

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title("3D Trajectories of Charged Particles")
    # plt.legend()
    # plt.show()

    # # 2D Plot of Energy vs Time
    # plt.figure()
    # energies = [
    #     sim._energy(loc[i, :, :], vel[i, :, :], edges) for i in range(loc.shape[0])
    # ]
    # plt.plot(energies)
    # plt.xlabel("Time")
    # plt.ylabel("Energy")
    # plt.title("Energy of the System Over Time")
    # plt.show()
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Function to plot trajectories and uniform force vector field
    def plot_trajectories_and_field(
        loc, force_field, box_size=10, grid_size=5, k=1.0, m=0.1
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot particle trajectories
        n_balls = loc.shape[-1]
        for i in range(n_balls):
            ax.plot(loc[:, 0, i], loc[:, 1, i], loc[:, 2, i], label=f"Particle {i+1}")
            ax.scatter(loc[0, 0, i], loc[0, 1, i], loc[0, 2, i], marker="o")

        # Create a grid of points within the box
        x = np.linspace(-box_size, box_size, grid_size)
        y = np.linspace(-box_size, box_size, grid_size)
        z = np.linspace(-box_size, box_size, grid_size)
        X, Y, Z = np.meshgrid(x, y, z)

        # Calculate force vectors at each point in the grid
        loc_grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        F_grid = force_field(loc_grid, k, m)

        # Plot the force vectors uniformly over the box
        ax.quiver(
            loc_grid[0, :],
            loc_grid[1, :],
            loc_grid[2, :],
            F_grid[0, :],
            F_grid[1, :],
            F_grid[2, :],
            color="r",
            length=2,
            normalize=True,
            alpha=0.6,
        )

        # Set plot limits
        ax.set_xlim([-box_size, box_size])
        ax.set_ylim([-box_size, box_size])
        ax.set_zlim([-box_size, box_size])

        # Labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Trajectories and Uniform Force Field of Charged Particles")

        # Add legend
        plt.legend()
        plt.show()

    # Call the function to plot the results
    plot_trajectories_and_field(
        loc, spiraling_conservative_field, box_size=10, grid_size=10
    )
