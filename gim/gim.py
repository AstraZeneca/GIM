import numpy as np 
import pandas as pd
import pickle

def calculate_coefficient_of_variance(df_dataset):
    """
    Function to calculate coefficient of variance. c_v = standard_deviation/mean.
    The dataset must be sliced based on the problem. 
    E.g. In TGG we calculate within each compound and in TCGA we use the entire dataset.
    
    Arguments: 
        df_dataset(pandas dataframe): Dataset to calculate c_v over
    
    Returns:
        top_values(list): c_v values listed in descending order
    
    
    """
    coeff_var = df_dataset.std()/df_dataset.mean()
    top_values = coeff_var.sort_values(ascending=False)
    return top_values

def harmonic_mean(x1, x2):
    """
    Calculate the harmonic mean between two values.
    """
    return (2*x1*x2)/(x1+x2)

def calculate_lower_upper_ratio(gene_i, gene_j, treatment_replicates, control_replicates, epsilon=1e-05):
    """
    Function to calculate the ratio of harmonic means of control and treatment in lower traingle and diagonal
    And ratio of control and treatment in upper traingle
    
    Arguments:
        gene_i (int): index of gene 1 
        gene_j (int): index of gene 2 
        treatment_replicates (numpy array): arrays of 'r' treatment replicates of 'n' genes. (n_t, n_g)
        control_replicates (numpy array): arrays of 'r' control replicates of 'n' genes. (n_c, n_g)
    Returns:
        lower_output (float64): replicate normalized harmonic mean ratio of gene_i vs gene_j
        upper_output (float64): replicate normalized ratio of gene_i vs gene_j
    """
    no_of_replicates_treatment = len(treatment_replicates)
    no_of_replicates_control = len(control_replicates)
    
    # Lower calculates ratio of harmonic means between treatment and control
    # Initial set to zero
    treatment_lower_sum = 0
    control_lower_sum = 0
    
    # Upper calculates ratio of ratio between treatment and control
    # Initial set to zero
    treatment_upper_sum = 0
    control_upper_sum = 0
    
    # Sum values in treatment replicates  
    for rt in range(no_of_replicates_treatment):
        # Refer Equation 3 numerator in Paper
        treatment_lower_sum += harmonic_mean(treatment_replicates[rt][gene_i], treatment_replicates[rt][gene_j])
        # Refer Equation 2 numerator in Paper
        treatment_upper_sum += treatment_replicates[rt][gene_i]/treatment_replicates[rt][gene_j]
    
    # Sum values in control replicates  
    for rc in range(no_of_replicates_control):
        # Refer Equation 3 denominator in Paper
        control_lower_sum += harmonic_mean(control_replicates[rc][gene_i], control_replicates[rc][gene_j])
        # Refer Equation 2 denominator in Paper
        control_upper_sum += control_replicates[rc][gene_i]/control_replicates[rc][gene_j]
    
    # Average the values based on number of replicates
    normalized_lower_treatment_sum = treatment_lower_sum/no_of_replicates_treatment
    normalized_lower_control_sum = control_lower_sum/no_of_replicates_control
    
    normalized_upper_treatment_sum = treatment_upper_sum/no_of_replicates_treatment
    normalized_upper_control_sum = control_upper_sum/no_of_replicates_control
    
    # Final value of lower is the natural log of ratio of harmonic mean of gene i and j 
    # Final value of upper is the natural log of ratio of ratio of gene i and j 
    # small epsilon value is added to the denominator
    # Refer Equation 1 in Paper
    lower_output = np.log(normalized_lower_treatment_sum/(normalized_lower_control_sum+epsilon))
    upper_output = np.log(normalized_upper_treatment_sum/(normalized_upper_control_sum+epsilon))
        
    return lower_output, upper_output 

def gim_transform(df_treatment, df_control):
    """
    This function takes the treatment and control files for a sample and returns a transformed GIP image-like matrix.
    If control does not exist, Take the mean of the entire dataset or use your own problem specific logic.
    Arguments:
        df_treatment (pandas dataframe): Dataframe containing treatment replicates. (n_t, n_g)
        df_control (pandas dataframe): Dataframe contraining control replicates. (n_c, n_g)
    
    Returns:
        transformed_mat (numpy array): Transformed GIP image-like matrix for CNN prediction. (n_g, n_g)
    """
    
    treatment_replicates = df_treatment.values
    control_replicates = df_control.values
    
    # Case where only there are no replicates 
    # Expand first dimension for computation 1 x n_g
    if df_treatment.ndim == 1:
        treatment_replicates = np.expand_dims(treatment_replicates, axis = 0)
    
    if df_control.ndim == 1:
        control_replicates = np.expand_dims(control_replicates, axis = 0)

    if control_replicates.shape[1] != treatment_replicates.shape[1]:
        print("Error: Please enter replicates with same gene set and shape")
    else:
        
        # Final size of the GIP matrix
        # Based on number of selected genes
        N = treatment_replicates[0].shape[0]
        transformed_mat = np.zeros((N, N))
        
        
        for gene_i in range(N):
            for gene_j in range(N):
                # lower_output is for the lower traingular (with diagonal) matrix - we use harmonic mean.
                # upper_output is for the upper traingular matrix - we use ratio here.
                lower_output, upper_output = calculate_lower_upper_ratio(gene_i, gene_j, treatment_replicates, control_replicates)
                
                if gene_i>=gene_j:
                    transformed_mat[gene_i, gene_j] = lower_output
                else:
                    transformed_mat[gene_i, gene_j] = upper_output
                    
        return transformed_mat