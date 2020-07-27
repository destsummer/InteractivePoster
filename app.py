# Imports
import os, flask, dash
from pathlib import Path
from random import randint
import dash_bootstrap_components as dbc
import gensim
from gensim.test.utils import datapath
from gensim.utils import simple_preprocess
import pyLDAvis
import pyLDAvis.gensim
import json
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import numpy

# Plots
from plotly.express import bar
import pandas as pd

# Import iPoster Object Class
from iposter.iposter import iPoster
import iposter.colors as colors

#*** Run Local Flag ***
RUN_LOCAL=False

# ******************Define Your Interactive Poster Here***************
# The following shows a sample interactive poster.
# Images for sections must be saved under the assets/ folder.
# You can import code from your own modules and construct the final dash
# interactive poster here.
def create_poster():

    # Instanitate an iPoster
    my_poster = iPoster(title="LDA Topic Modeling for Suicide Prevention", # Title of your poster
                        authors_dict={"Destinee Morrow" : ["Hood College", "Lawrence Berkeley National Laboratory"], # Authors in {student, mentors, PI} order
                                      "Rafael Zamora-Resendiz" : ["Lawrence Berkeley National Laboratory"],
                                      "Shirley Wang" : ["Lawrence Berkeley National Laboratory"],
                                      "Xinlian Liu" : ["Hood College", "Lawrence Berkeley National Laboratory"],
                                      "Silvia Crivelli" : ["Lawrence Berkeley National Laboratory"]},
                        logo = "hood.png", # Home institution logo
                        banner_color=colors.DOE_GREEN, # Color of banner header; colors has preset colors
                        text_color=colors.WHITE)

    # Add sections to first column then add new column
    my_poster.add_section(title="Abstract",
        text="Suicide is a major national health concern. Over the last twenty years, the suicide rate increased by 50% in the US. Among all groups, veterans have the worst suicide rate, almost twice as high as the general population, even after the normalization of age and gender. Current theories in mental health studies failed to create predictive models that can be used in suicide prevention. With the main barriers being the lack of publicly available data and the lack of sophisticated methods able to extract the socio-economic determinants from the large, complex and noisy data. Adoption of Electronic Health Records (EHR) has led to a sharp increase in clinical data in forms of structured data, such as demographic information, diagnosis, medication, lab results, billing codes, etc, and unstructured data such as free format physician notes. These notes often include rich information that is not usually captured by structured data. Processing, formatting, and parsing of such notes are essential in devising machine learning algorithms to produce clinical decisions. Latent Dirichlet Allocation (LDA) is a generative statistical model that allows us to find topics in such free format text notes. Can we create a LDA model that identifies topics related to the determinants of suicide? Working with doctors at the U.S Department of Veterans Affairs (VA), we identified a group of patients with billing codes related to Diseases of Despair (DoD). This group represents an expansion to the suicide group to include drug abuse, alcoholism and other high-risk patients. We applied LDA to an Intensive Care Unit (ICU) EHR data set and looked for topics in the notes. Our models are sensitive enough to pick up different topics between genders. Results of suicide attempt patient notes are also presented for comparison and validation.",
    	img0={"filename":"VA_rate.png", "height":"5in", "width":"8in", "caption":"Veteran Suicide Annual Report, September 2019, https://www.mentalhealth.va.gov/docs/data-sheets/2019/2019_National_Veteran_Suicide_Prevention_Annual_Report_508.pdf"})
    my_poster.add_section(title="Background",
        text="Social determinants such as homelessness, hopelessness, and social isolation are important contributors to an elevated risk of suicide. Natural Language Processing (NLP) of the unstructured data has succeeded in producing insights in medical applications involving such concepts in addition to major diagnoses. However, action plans were not implemented. Our methods and findings will be incorporated into our work with the VA to generate results for decision making to help improve responsiveness.")
    my_poster.add_section(title="MIMIC-III and Diseases of Despair",
        text="The MIMIC-III [1] data is curated by MIT. It includes approximately 60,000 ICU admissions to Beth Israel Deaconess Medical Center in Boston, MA from June 2001 to October 2012. Patients with DoD International Classification of Diseases (ICD) codes are used as the focus of our study. The DoD group includes codes related to suicide risks, such as sleep disorders, migraine, schizophrenic disorders, mood disorders, anxiety, personality disorders, post-traumatic stress disorder (PTSD), depression, among others.",
        ref= "1. MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available at: http://www.nature.com/articles/sdata201635")
    my_poster.next_column()

    # Add sections to second column then add new column
    my_poster.add_section(title="Methods ",
        text="Preprocessing of the text includes stemming and lemmatization of notes. Then we create a dictionary and convert it to Bag of Words (BOW). The weight of each word is calculated using Term Frequency-Inverse Document Frequency (TF-IDF). The LDA model is trained with BOW and TF-IDF. We inspected topics generated by the model and validated them by reading the actual notes. We trained the models with all available notes for the maximized information, and also with a specific section on the patient’s history. Looking directly at the patient history allows for us to analyze the past, family and social histories of patients which tends to have more socio-economic information that we are interested in.")
    my_poster.add_section(title="Figures",
        img1={"filename":"Topic_5_history.html", "height":"5.25in", "width":"10in", "caption":"Example topic when processing only the history section of discharge notes for both females and males. This topic is related to cancer, indicated by top weighted words 'cancer', 'lung' 'cell', 'mass', 'metastatic' and 'carcinoma'. For each of these words in this topic, total word count and word weight is projected."},
        img2={"filename":"Topic_9_male2.html", "height":"5.25in", "width":"10in", "caption":"Example topic when processing only the history section of discharge notes for male patients. This topic is related to the heart and Coronary artery disease indicated by top weighted words 'aortic', 'cardiac' 'coronaries', 'disease', and 'valve'. For each of these words in this topic, total word count and word weight is projected."},
        img3={"filename":"SA_History_dis.html", "height":"5.25in", "width":"10in", "caption":"This graph shows the frequency of dominant topics and weighted topics found within suicide attempt patient notes when evaluated against the LDA model trained on the history section of discharge notes. Dominant topic means that a note may have a probability for matching 3 topics but only the highest is recorded. Topic weightage means that all probabilities for notes are accumulated and recorded by topic. Reference Image 5 for more in depth information regarding the specific topics."})
    my_poster.next_column()

    # Add sections to third column then add new column
    # Visualize the topics and words
    
    my_poster.add_section(title="Interactive LDA Plot",
    	pyLDA={"filename":"full_lda_html.html", "height":"7in", "width":"12in", "caption": "Scroll over the various topics (1-20) to inspect the corresponding words including their frequency and weight. This LDA model was trained using only the history section of discharge notes for both male and female patients. Multidmensional Scaling (MDS) gives an estimate of similarity between the topics. Relevance metric can be adjusted to effect saliency and relevance."})
    my_poster.add_section(title="Findings", text="The LDA model trained on the history section of discharge notes was able to pull out more specific social determinants that were unrelated to their ICU stay, compared to the LDA trained on all available notes. When evaluating how the suicide attempt group, our validation group, matched to the topics produced by LDA model using all available notes, it can be seen that many of the patients experienced symptoms related to withdrawl, were scored using the Clinical Institute Withdrawal Assessment for Alcohol (CIWA), had mentions of various bodly ailments, and more. The LDA model trained using the history notes identified that the suicide attempt group had histories involving alcoholism, hypertension, heart disease, cancer and more. To narrow this down even further, the history LDA model was trained again after separating male and female patients. Female suicide attempt patients had histories of hypertension, drug abuse, depression and more, whilst male suicide attempt patients had histories of coronary artery disease, alcohol abuse, cirrhosis and hopelessness.")
    my_poster.add_section(title="Conclusion", text="NLP allows for the extraction of concepts and topics that are hidden within EHR free text data. These LDA models were successful at identifying topics related to diagnosis, procedures, routines and more. These LDA models were also successful at identifying socio-economic determinants like drug abuse. Using these topics and incorporating the top weighted words found in the notes, future implementation on the CDW of the VA can identify those who are at higher risk of suicide. Further development and refining is ongoing.")
    my_poster.add_section(title="Acknowledgments", text="This work was supported in part by the U.S. Department of Energy, Office of Science, Computational Research Division (CRD) of the Berkeley National Lab, and VA Million Veteran Program (MVP). Thank you again to Dr. Liu, Dr. Crivelli, Rafael, Shirley and to all the other group members for making this project memorable.") 
    my_poster.next_column()

    return my_poster.compile()

# **********************************************************************

# Dash App Configuration
if RUN_LOCAL:
    app = dash.Dash(__name__,
                    assets_folder= str(Path(__file__).parent.absolute())+"/assets",
                    assets_url_path='/',
                    external_stylesheets=[dbc.themes.BOOTSTRAP],
                    suppress_callback_exceptions=True)
else:
    server = flask.Flask(__name__)
    server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
    app = dash.Dash(__name__,
                    server=server,
                    assets_folder= str(Path(__file__).parent.absolute())+"/assets",
                    assets_url_path='/',
                    external_stylesheets=[dbc.themes.BOOTSTRAP],
                    suppress_callback_exceptions=True)
app.layout = create_poster()

# Main Function
if __name__ == "__main__":
    if RUN_LOCAL:
        app.run_server(debug=False, host="0.0.0.0", port="8888")
    else:
        app.server.run(debug=True, threaded=True)


