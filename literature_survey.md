# Literature Survey: DGA Detection Methodologies

## 1. Traditional blacklisting and Signature Matching
Historically, defending against Domain Generation Algorithms (DGAs) relied on reactive blacklisting (such as updating DNS blocklists) and signature matching of known malware executables. Security operations centers maintained lists of suspected domains and updated firewalls and DNS sinkholes daily. 

However, this approach fails against modern modular malware. Cybercriminals utilize dynamic seeds (like hot temperature indexes, foreign exchange rates, or daily news headers) to generate thousands of pseudo-random domains daily. Because only a single domain needs to resolve to establish a Command and Control (C2) channel, defender blacklists are perpetually outdated.

## 2. Classical Machine Learning on Lexical Features
To overcome the limitations of static signatures, researchers shifted toward classifying domains based on their structural characteristics (lexical features). 

By extracting statistical metrics such as domain length, Shannon entropy of character distributions, digit densities, and vowel-to-consonant ratios, researchers trained classifiers like Support Vector Machines (SVM), Decision Trees, and Random Forests. Later, the extraction of character bigrams and trigrams (character n-grams) was introduced to capture the probability of character transitions typical of natural language (such as "th" or "ing" in English).

While classical ML algorithms are highly efficient and run at network speed, they have a key limitation: they treat features as independent counts and cannot capture the sequential context of characters.

## 3. Deep Learning and Sequential Architectures
In recent years, Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures, have been applied to DGA detection. 

These models process domain names as character sequences rather than static feature tables. By training embedding layers to map characters to dense vector spaces, sequence models learn character transition rules natively. LSTMs successfully identify random vowel/consonant clustering even when malware authors try to mimic natural language (dictionary-based DGAs).

The bottleneck of sequential deep learning models is execution speed. Running neural embedding and recurrent cell calculations on CPU-based network gateways introduces latency bottlenecks, which are challenging for real-time intrusion detection systems.

## 4. Hybrid Stacking and Ensemble Learning
The state-of-the-art in DGA research is moving toward hybrid models. Stacking ensembles use a deep sequence model (like an LSTM) as a feature generator, feeding its soft prediction probability into a tree-based classifier (like XGBoost or LightGBM) alongside classical lexical statistics. 

This hybrid stacking architecture provides a Pareto-optimal compromise: it leverages the sequence intelligence of deep learning models while maintaining the lightweight, parallel execution speeds of gradient-boosted trees.
