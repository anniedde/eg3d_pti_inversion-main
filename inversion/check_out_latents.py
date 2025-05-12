import numpy as np
import os
import matplotlib.pyplot as plt

vector_norms = []  # List to store the norms of all vectors

for t in range(1):
    directory = f'/playpen-nas-ssd/awang/data/Margot/{t}/train/preprocessed'
    file_list = [file for file in os.listdir(directory) if file.endswith('.npy') and 'latent' in file]

    vectors = []
    latent_shape = np.load(os.path.join(directory, file_list[0])).shape
    for file in file_list:
        file_path = os.path.join(directory, file)
        vector = np.load(file_path).flatten()
        vectors.append(vector)
        vector_norm = np.linalg.norm(vector)
        vector_norms.append(vector_norm)  # Append the norm to the list

    average_norm = np.mean(vector_norms)  # Calculate the average norm
    print('Average norm of all vectors:', average_norm)

    mean_vector = np.mean(vectors, axis=0)
    

    # find the distance of each vector from the mean vector
    distances = []
    directions = []
    for vector in vectors:
        distance = np.linalg.norm(vector - mean_vector)
        distances.append(distance)
        
        direction = vector - mean_vector
        print('norm of direction: ', np.linalg.norm(direction))
        directions.append(direction)
    average_direction = np.mean(directions, axis=0)
    print('norm of average direction: ', np.linalg.norm(average_direction))
    
    average_distance = np.mean(distances)
    variance_distance = np.var(distances)
    print('Average distance:', average_distance)
    print('Variance of distances:', variance_distance)
    print('standard deviation of distances:', np.std(distances))

    directions_matrix = np.column_stack(directions)
    # get svd
    u, s, vh = np.linalg.svd(directions_matrix)
    plt.bar(range(len(s)), s)
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Values of Directions Matrix')
    plt.savefig('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/singular_values.png')
    plt.show()

    out_dir = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/out/temp'

    for j in range(100):
        # get random direction vector in shape of mean_vector
        direction = np.random.normal(size=mean_vector.shape)
        direction /= np.linalg.norm(direction)
        # draw from normal distribution with mean of average distance and std of 0.1
        direction *= np.random.normal(loc=average_distance, scale=np.std(distances))
        #direction *= average_distance
        u_vec = (mean_vector + direction).reshape(latent_shape)
        np.save(os.path.join(out_dir, f'latent_{j}.npy'), u_vec)

    print('Mean distance:', average_distance)
    print('Variance of distances:', variance_distance)

    
    