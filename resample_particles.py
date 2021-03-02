def resample_particles(particle_poses, particle_weights, num_particles):
    # particle_poses = particle poses
    # particle_weights = particle weights
    # num_particles = number of particles

    # keep count of total number of particles
    count_num_particles = 0

    # iterate through all particles
    particle_weight = particle_weights[0, 0]

    # initialize variables to store resampled particle poses and weights
    resampled_particle_poses = np.zeros((3, num_particles))
    resampled_particle_weights = np.tile(1 / num_particles, num_particles).reshape(1, num_particles)

    for i in range(num_particles):
        # intitilize a uniform distribution fo weights based on number of particles
        uniform_distribution = np.random.uniform(0, 1 / num_particles)
        # particle weights with uniform distribution
        uniform_particle_weight = uniform_distribution + i / num_particles
        # if uniform particle weight is greater than the current particle weight, add more weight to
        # the particle until it has greater weight than the initialized base uniformly distributed weights
        while uniform_particle_weight > particle_weight:
            count_num_particles += 1
            particle_weight = particle_weight + particle_weights[0, count_num_particles]

        # resize particle_poses storage variable based on number of particles
        resampled_particle_poses[:, i] = resampled_particle_poses[:, count_num_particles]
    return resampled_particle_poses, resampled_particle_weights
