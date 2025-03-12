Blog
 / 
TutorialsConcrete ML
Build an End-to-End Encrypted 23andMe-like Genetic Testing Application using Concrete ML
July 17, 2024
  -  
Andrei Stoian
 One challenge of the Zama Bounty Program Season 5 was to create an encrypted version of 23andMe (or other DNA testing platform) using Fully Homomorphic Encryption (FHE). Github user Alephzerox and Github user Soptq successfully completed this bounty, and this blog post is based on their contributions. Curious about the Zama Bounty Program? Learn more on Github.


Over 30 million people have taken DNA tests to determine their ancestry through computer genetic genealogy. By processing the digitized sequences of DNA bases, sophisticated computer algorithms can identify if one’s ancestors came from a number of population groups determined by a common geographic origin of ancestors. DNA is sensitive personally identifiable information (PII) as it can identify an individual uniquely, and leaks of DNA data have already happened.

The Zama Bounty Program awards developers who build applications that showcase the value FHE brings to protecting sensitive data, and DNA is one of the most important pieces of PII that needs protection. In Season 5 of the Zama Bounty Program, we challenged our community to build a machine learning system that determines ancestry on encrypted DNA data.

Two solutions shared the first prize and are discussed in this blog: one by Github user alephzerox and the other by Github user soptq.

The two winners implemented different strategies but share a similar data pipe-line. Both solutions used DNA data from the 1000 genomes project. Humans have 23 pairs of chromosomes, packages of DNA genetic material, and the solutions presented use data from chromosome 22. For the purposes of these algorithms, a chromosome is represented as a vector of 1's and 0’s where each such binary value indicates the genetic variation at a position in the chromosome. Such values are called single nucleotide polymorphisms (SNPs).

Let's take a look at the first solution: Using ML to predict ancestry
The approach taken by Github user soptq follows Gnomix [1] and inspects 517 windows - intervals of the DNA description vector - using logistic regression predictors to determine per-window ancestry. In a second step, the decisions on these windows are aggregated using a second classifier which determines the person’s ancestry mixture.

Below is an example of genome data. In total the dataset contains 4480 individuals and the chromosome that is used has 1,059,079 SNPs.

# Training set individuals
[1 0 1 1 0 1 0 0 0 1 0 0 ... 1]
[1 0 1 0 1 1 0 0 0 1 1 0 ... 0]
In a pre-processing step, training data is generated based on “pure-blooded” genomes called “founders”. These are genomes to which an ancestry was assigned manually and are considered to have a single ancestry. To obtain training data, “admixing” simulation is performed: random pairs of parent genomes are combined in random proportions  and synthetic labels - the ancestry proportions contained in the generated genome - are computed. 

To start, random break points in the 22nd chromosome SNP vector are generated.

num_snps = 10  # Total number of SNPs in the chromosome

breakpoints = np.random.choice(
range(1, num_snps), 
size=int(sum(np.random.poisson(0.75, size=gen))) + 1,  # Number of breakpoints
replace=False  # No repeated breakpoints
)

Next, SNPs are copied from one parent to another at these random break-points, using one chromosome of the 22nd pair:

# Parent 2: Choose one random chromosome of the 22nd pair
snp, label = (snp_1, label_1) if random.random() < 0.5 else (snp_2, label_2)

...

# Parent 2: Choose one random chromosome of the 22nd pair
_snp, _label = (_snp_1, _label_1) if random.random() < 0.5 else (_snp_2, _label_2)

# Copy SNPs from parent1 to the "admixed" offspring of parents 1 & 2
snp[breakpoints[i]:breakpoints[i + 1]] = 
_snp[breakpoints[i]:breakpoints[i + 1]].copy()

# Generate the synthetic ancestry label
label[breakpoints[i]:breakpoints[i + 1]] 
= _label[breakpoints[i]:breakpoints[i + 1]].copy()
Once training data is generated, it is split into three sets: first stage training set, second stage training set and validation set. To start, Github user soptq trained Logistic Regression models for the first stage of the classification pipe-line, for each position in the chromosome. As shown in the snippet below, Concrete ML classifiers are a plug-in replacement for scikit-learn ones, which eases development.

from concrete.ml.sklearn import LogisticRegression

# Compute the number of windows in the chromosome
n_windows = chromosome_length // window_size 	# number of windows
context = int(window_size * 0.5) 			# overlap between windows

# Initialize Concrete ML Logistic Regression models 
# There will be one model for each window position in the chromosome
# Training is done on clear data
base_models = [LogisticRegression(n_bits=8, 
penalty="l2", 
C=3., 
solver="liblinear", 
max_iter=1000) for _ in range(n_windows)]

# Extract windows with some context overlap
padded_window_size = window_size + 2 * context
# Starting indices of each window
idx = np.arange(0, chromosome_length, window_size)[:-2]
# Extract windows from the first stage training set X_t 
# to create training data for each window classifier
X_b = np.lib.stride_tricks.sliding_window_view(
X_t, 
padded_window_size, 
axis=1)[:, idx, :]

# Train models for each of the windows in the chromosome
models_with_data_and_labels = tuple(
zip(models[:-1], 
np.swapaxes(X_b, 0, 1), 
np.swapaxes(y_t, 0, 1)[:-1])
)
        
for (model, x, y) in tqdm(models_with_data_and_labels):            
    model.fit(x, y)
    model.compile(x)
The second stage of this approach predicts the global ancestry for a chromosome based on the predictions of the first stage. To train the second stage, the predictions of the first stage classifiers must be computed first, on a second training set split.

# Sliding window extraction of the second training split, X_p
X_b = np.lib.stride_tricks.sliding_window_view(X_p, padded_window_size, axis=1)[:, idx, :]
models_and_1st_stage_proba = tuple(zip(models[:-1], np.swapaxes(X_b, 0, 1)))

# Use prediction in the clear to produce training data for the second stage
prob_X_t2 = np.array([model.predict_proba(x, fhe="disable") 
for (model, x) in models_and_1st_stage_proba]
)
Next, the second stage classifier is trained and compiled. A quantization of 4 bits was used, which is the optimal setting for tree-based models, and the max depth of the trees was set to 4 to avoid overfitting.  

from concrete.ml.sklearn import XGBClassifier

smoother = XGBClassifier(	
     n_bits=4, n_estimators=100, max_depth=4,
	learning_rate=0.1, reg_lambda=1, reg_alpha=0,
	n_jobs=N_JOBS, random_state=SEED,
	use_label_encoder=False, objective='multi:softprob',
)
   
X_slide, y_slide = slide_window(prob_X_t2, 75, y_t)
smoother.fit(X_slide, y_slide)
smoother.compile(X_slide, p_error=P_ERROR)
Concrete ML classifiers, when trained on clear data as is the case in this work, use scikit-learn training algorithms under the hood. Thus all hyperparameter settings that data-scientists are familiar with are supported, as shown in the code above. 

The accuracy of Github user soptq method reaches 96% and the latency of inferring ancestry for an individual’s encrypted genome is ~300s.

A closer look at the second solution: Similarity search in a reference panel of genomes
A second solution proposed by Github user alephzerox implements the SALAI-Net paper [2] and relies on a reference panel of individuals. This set of genomes, which contains "pure blooded" individuals, is annotated with the ancestries that the application wants to identify. The query chromosome is the sensitive information that must be protected and is encrypted, while the reference panel can be kept in the clear. 

The first step of the algorithm is to count the number of common SNPs between the query chromosome and each reference panel chromosome.  Converting values of 0 in the SNP vectors to -1, this first step is performed by using a multiplication between each query chromosome SNP vector and each reference panel chromosome, followed by a convolution that aggregates the matches.

from concrete import fhe

reference_panel = self._reference_panel		     # Clear reference panel
snp_count = self._active_batch_samples.shape[1]     # No. snps per chromosome
window_size = self._model_parameters.window_size    # Analysis window size
population_count = reference_panel.population_count # Size of reference panel

# ------------ Compute SNP matches ------------
samples_slice = self._active_batch_samples
snp_matches = snps * samples_slice	# Multiply to check matches

# ------------ Compute window similarity scores ------------
snp_matches_reshaped = snp_matches.reshape(1, 1, population_count, snp_count)

sum_kernel = np.array([[[[1] * window_size]]])
window_similarity_scores = fhe.conv(snp_matches_reshaped, sum_kernel, strides=(1, window_size))
In a second step a “smoother” kernel is applied on the raw similarity scores in the sliding  window fashion. The kernel has a wave-like shape which was learned in the SALAI-Net paper [2].

per_population_scores = per_population_scores.reshape(population_count, 1, window_count)

smoother_kernel = inference_task.model_parameters.smoother_weights_as_tensor

smoother_kernel_size = len(smoother_kernel)
smoother_kernel = smoother_kernel.reshape(1, 1, smoother_kernel_size)

smooth_scores = f.conv1d(per_population_scores, smoother_kernel, padding=smoother_kernel_size // 2)
In each window of the query chromosome we thus obtain similarity scores to the reference panel chromosome strand in that window. The top-1 reference chromosome is taken as the label associated with a specific window. Finally, the percentage of each ancestry is computed from the frequencies of each reference panel ancestry that were predicted.  In the submitted approach the top-1 computation is not performed in FHE, though it is easy to implement with the max-pool operator from the Zama Concrete library.

The accuracy of the Github user alephzerox method is 96% on the test set of the 1000 genomes dataset, provided the reference panel contains 40 founders for each ancestry. The accuracy is proportional to the number of founders per ancestry:

Founder	Accuracy
1	29%
5	50%
10	66%
40	96%
For this setting, with 40 founders, FHE latency should be on the order of tens of minutes on a large 192-core machine. 

Conclusion
Both solutions achieve good accuracy for ancestry classification. While taking different approaches, the latency complexity between the two solutions is similar, with the second solution performing many more linear computations: multiplication with scalars and convolution.  With respect to non-linear computations that require PBS, the XGBoost classifier used in Github user soptq solution has similar complexity in FHE as the top-1 computation needed for Github user alephzerox method. Overall, while both methods obtain similar accuracy, but using machine learning as in Github user soptq approach gives lower inference latency on encrypted data.