# Combating Employment Scams: True and Fake Job Classification with NLP
## Project Overview
In today's competitive job market, navigating through countless online postings can be daunting. Unfortunately, amidst the genuine opportunities lurk insidious employment scams, putting job seekers at risk of financial loss and identity theft. These scams have skyrocketed, shockingly doubling cases in 2018 alone. This alarming trend calls for a proactive solution to safeguard vulnerable individuals. This project tackles this critical issue head-on, aiming to empower job seekers with a cutting-edge AI-powered classifier built with the help of Machine Learning and Natural Language Processing (NLP). Our innovative system will accurately distinguish between genuine and fraudulent job advertisements, shielding users from the deceptive tactics of scammers. By analyzing linguistic patterns and leveraging advanced algorithms, we strive to create a secure and trustworthy online job search experience, ultimately protecting individuals from falling victim to these malicious schemes.

## Dataset
**Origin**: <a href = "http://emscad.samos.aegean.gr/"> University of the Aegean </a> <br>
**Rows**: 17,880

**Sythetic Dataset**: <a href = "https://gretel.ai/"> Gretel.ai </a><br>
**Rows**: 5000

## Initial Data Analysis
### **Education Level**

The distribution of educational qualifications in our dataset reflects a broad spectrum, encompassing qualifications from Bachelor's Degrees to diverse vocational certifications. **Notably, we observed an intriguing pattern during the analysis of education data. Despite similar educational requirements across multiple job postings, distinct labels were assigned to them. For instance, some job listings specify a requirement for a "Bachelor's Degree," while others opt for the term "Bachelor's or Equivalent." This discrepancy underscores the need for a meticulous approach to data cleaning and mapping to standardize and unify similar qualifications, ensuring a consistent representation for accurate analysis and model training.**

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/7eadf0dd-c4a2-43b6-82a7-97e6d60e3cfa)


### **Industry Level**

In our dataset, we observed a diverse array of job postings spanning various industries. Notably, Information Technology and Services emerged as the dominant sector, constituting 11.3% of the total job postings. The pie chart below represents and highlights the top 10 industries. Additionally, job postings from industries outside this top list have been amalgamated into the "Others" category for a more concise and clear presentation of the data.

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/f2445ea1-7b95-499f-bdfd-f01d6bc27603)


## Data Cleaning and Preprocessing Overview

In the pursuit of refining our dataset, several key steps were undertaken to ensure optimal quality and readiness for analysis.

### Data Cleaning

-  **Data Augmentation**: We bolstered the dataset's diversity and volume by integrating synthetic data, enhancing the model's training and generalization capabilities.
-  **Handling Null Values**: Null values in specific columns were replaced with 'Unspecified,' excluding certain columns where null values were retained as NaN.
-  **Mapping Education Level**: A custom function was applied to standardize education labels, unifying similar qualifications for a more streamlined analysis.
-  **Location Standardization**: We introduced a dedicated column, extracting two-character country codes from the inconsistent location field, facilitating standardized analysis based on country codes.
-  **Industry Mapping**: A custom function was applied to reorganize industries in the job description based on the common traits for a more streamlined analysis.
-  **Number to Text Transformation**: Binary values in job-related columns were mapped to descriptive labels, enhancing clarity for telecommuting and work-from-home permissions.
-  **Merging Text Columns**: Once data was cleaned, all text columns were consolidated into a unified 'full_text' column, excluding numeric columns, to enhance the efficiency of textual analysis.

### Text Processing

-  **Tokenization**: Full text was segmented into individual words to facilitate further analysis.
-  **Lowercasing**: Consistency was ensured by converting all text to lowercase.
-  **Removal of Non-Alphabetic Characters**: Non-alphabetic characters were filtered out to retain only meaningful words.
-  **Stop Word Removal**: Common English stop words were eliminated to enhance the relevance of the text.
-  **Lemmatization**: Words were reduced to their base or root form for improved language consistency.
-  **Part-of-Speech Tagging**: Grammatical categories were assigned to each word, offering insights into syntactic structure.
-  **Detokenization**: Preprocessed tokens were reconstructed into coherent text for further analysis.
-  **Vectorization**: Various methods, including TF-IDF, count vectorization, word2vec, and BERT, were implemented to convert text into numerical vectors, capturing semantic relationships and context for improved data representation.

### Improved Data Representation
Vectorization methods facilitated compatibility with a wide range of machine learning and neural network models, enhancing flexibility and model performance by providing meaningful numerical representations of textual data.

## Exploratory Data Analysis (EDA)
1.  **Employment Type**
The 'employment_type' column predominantly consists of 'Full-time' positions with 14,931 entries, followed by 'Contract' with 1,762, 'Part-time' with 1,054, 'Other' with 308, and 'Temporary' with 266 entries.

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/c75629af-b620-4de5-94bc-df24ef4c310f)


2.  **Locations (Country)**

From the total job postings, 64% of the job postings mentioned are from the United States, followed by Great Britain with 10.5%

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/f1ea231d-86d8-46b3-8650-682d5228468d)


4.  **Industries** 
Before cleaning and mapping, the dataset contained various industry categories, including Customer Service, Information Technology, Engineering, Sales, Administrative, Marketing, Accounting/Auditing, Other, Design, and Management. However, these categories lacked uniformity and clarity. After implementing the cleaning and mapping process, the industries were reorganized based on their common traits, resulting in a more cohesive and standardized representation. The categories were consolidated into broader groups such as Technology, Finance, Energy, Marketing, Education, Services, Healthcare, Telecommunications, and Property. This restructuring facilitated a clearer understanding and comparison of industry roles, streamlining the analysis process and improving the dataset's overall organization and usability.

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/8fa11cc2-c474-45d7-afe8-0549446568f5)



4.  **Educational Qualifications**
Prior to the cleaning and mapping process, the educational qualifications in the dataset exhibited a diverse range, including Bachelor's Degrees, High School or equivalent, Unspecified, Master's Degrees, Associate Degrees, Certifications, Some College Coursework Completed, Some High School Coursework, Professional qualifications, Vocational training, Doctorates, Vocational - HS Diploma, and Vocational - Degree. Following the cleaning and mapping steps, these qualifications were standardized and grouped into broader categories. The refined classifications now include Bachelor, High School, Unspecified, Master, Associate, Certification, Some College, Professional, Vocational, and Doctorate. This restructuring simplifies the representation of education levels, providing a clearer and more consistent foundation for subsequent analysis and modeling.

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/a6da2348-d1f3-4d41-8f11-6350faa3829b)


5.  **Experience**
 The job postings in the dataset exhibit a diverse range of experience requirements. The most frequently mentioned experience levels include the Mid-Senior level, with 4307 occurrences, followed closely by Entry level at 4130. This distribution reflects the varied experience expectations across the job listings in the dataset.

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/2d4ebeb4-5823-4527-b2a5-8ca38127c617)


6.	**Numerical Features Correlation Analysis**
The correlation matrix highlights key associations, particularly concerning the 'fraudulent' variable. Job postings with company logos show a significant negative correlation (-0.56) with fraudulence, suggesting that such postings are less likely to be fraudulent. Similarly, the presence of questions in job postings exhibits a moderate negative correlation (-0.27) with fraudulence, indicating that postings with questions are also less prone to fraudulent activity. Conversely, the 'job_id' variable shows a moderate positive correlation (0.38) with fraudulence, suggesting a potential association.

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/b6b1fab5-2af8-4dc2-aa6a-75f2cdf9b9b1)


7.	**Bigram Analysis: Unveiling Linguistic Patterns in Job Postings**

In exploring bigrams extracted from genuine and deceptive job postings, discernible patterns emerge both before and after text cleaning. Prior to cleaning, authentic job postings exhibited commonplace phrases like "UnSpecified" and "present," pointing to potential ambiguities in information. After cleaning, prevalent bigrams shifted towards specific skills and qualifications, emphasizing terms such as "customer service," "communication skill," and "information technology." In contrast, deceptive job postings featured prominent bigrams like "UnSpecified" and "absent" before cleaning, suggestive of potential vagueness or misinformation. Post-cleaning, notable bigrams included "data entry," "entry-level," and "training provided," indicating a potential focus on entry-level positions with training opportunities. The transformed bigrams offer valuable insights into the linguistic nuances of authentic and deceptive job postings.

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/ce889a84-6a69-43ee-b13a-35b3fcf91aed)
![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/3570a267-4314-48cf-8a2a-e104a90a4406)

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/fedb9db8-64d8-47ea-af55-14da4535b399)
![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/a84bd290-7c41-490c-b231-20195490fbd8)

8. **Industry Disparities in Genuine and Deceptive Job Postings**
When examining the top industries in both real and fake job postings, some notable patterns emerge. In authentic job postings, the technology sector dominates with 4254 job posts, followed by finance, education, and marketing. On the other hand, fake job postings present a different landscape, with energy taking the lead at 1146 posts, followed by finance and technology. Interestingly, while technology is a significant sector in both cases, its prominence shifts between real and fake postings. 

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/479a75a5-15d2-4b5f-a9e8-3099ec8f8fc5)

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/7415042e-34fa-4243-9525-991cffe904c4)



9. **Education Requirements for Genuine and Deceptive Job Posts**

In genuine job postings, the majority of positions require qualifications categorized as "Other," indicating a diverse range of educational backgrounds. High school education is also commonly sought after, followed by bachelor's degrees. However, in deceptive job postings, the trend shifts significantly, with the most common requirement being a bachelor's degree, followed closely by high school education. The prevalence of unspecified qualifications is notably higher in fake job postings compared to genuine ones. Additionally, while master's degrees and certifications are still sought after, their frequency is considerably lower in fake job postings. The data suggests that deceptive job postings often target individuals with lower educational qualifications, potentially exploiting vulnerabilities in the job market.

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/331b93fa-c6bb-436f-a356-e30d85bd3a25)

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/f8aff7be-501f-4d50-81fc-9d96f9d4d25f)

## Splitting Data into Train-Test
We employed the train_test_split function to partition the dataset into training and testing sets, both for the original cleaned full-text data and the cleaned tokenized text data with a test size of 40%.

## Modeling and Vectorization Overview
In our project, we employed a diverse set of techniques for modeling and vectorization to comprehensively analyze and tackle the complexities of employment scam detection. Our vectorization methods included TF-IDF, Count Vectorization, and Word2Vec, each offering unique perspectives on textual data. In conjunction, we applied various machine learning algorithms such as Naive Bayes, Random Forest, and SVM, leveraging their strengths to create a robust and adaptable model. Furthermore, we explored the capabilities of recurrent neural networks (RNN) and long short-term memory (LSTM) models, harnessing the power of deep learning for nuanced pattern recognition. The evaluation process involved generating detailed classification reports, providing insights into the performance and effectiveness of each approach. This multifaceted strategy aimed to enhance the overall accuracy and reliability of our employment scam detection system.

![image](https://github.com/doshiharmish/Combating-Employment-Scams-True-and-Fake-Job-Classification-with-NLP/assets/16878994/683f3704-3747-4451-8ae9-f5706e95baf1)

## Conclusion
In our project to classify real and fake jobs, we employed various machine learning models. All of the models achieved satisfactory performance on the classification task, with Naive Bayes, Random Forest, and SVM achieving accuracies between 97% and 98%. However, the classification reports indicated that these models were overfitting the data, as they exhibited high precision and recall scores on the training data but lower scores on the validation data. To address this overfitting issue, our group switched to neural network models, namely RNN and LSTM. These models demonstrated superior performance on the validation data, achieving accuracies of 99%. This suggests that RNN and LSTM models possess a better ability to generalize to unseen data and are less susceptible to overfitting.

## References:
-  https://www.ijraset.com/best-journal/fake-job-detection-using-machine-learning
-  https://www.researchgate.net/publication/371508732_Fake_Job_Detection_with_Machine_Learning_A_Comparison
-  https://turcomat.org/index.php/turkbilmat/article/view/13533
-  https://www.researchgate.net/publication/360849325_A_machine_learning_approach_to_detecting_fraudulent_job_types
-  https://www.hindawi.com/journals/js/2022/4583512/
-  https://www.trendytechjournals.com/files/issues/volume6/issue1-3.pdf
-  https://link.springer.com/article/10.1007/s11063-021-10727-z
