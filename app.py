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
                                      "Xiange Wang" : ["Lawrence Berkeley National Laboratory"],
                                      "Xinlian Liu" : ["Hood College", "Lawrence Berkeley National Laboratory"],
                                      "Silvia Crivelli" : ["Lawrence Berkeley National Laboratory"]},
                        logo = "hood.png", # Home institution logo
                        banner_color=colors.DOE_GREEN, # Color of banner header; colors has preset colors
                        text_color=colors.WHITE)

    # Add sections to first column then add new column
    my_poster.add_section(title="Abstract",
        text="Suicide is a major national health concern. Over the last twenty years, the suicide rate increased by 50% in the US. Among all groups, veterans have the worst suicide rate, almost twice as high as the general population. Adoption of Electronic Health Records (EHR) has led to a sharp increase in clinical data in forms of structured data, such as demographic information, medications, billing codes, etc, and unstructured data such as free format physician notes. These notes often include rich information that is not usually captured by structured data. Processing, formatting, and parsing of such notes are essential in helping produce clinical decisions. Latent Dirichlet Allocation (LDA) is a generative statistical model that allows us to find topics in such free format text. Preliminary work showed that our LDA models are sensitive enough to pick up different topics related to determinants of suicide between genders. Currently, we are working with doctors at the U.S Department of Veterans Affairs (VA) and implementing a secondary natural language processing (NLP) tool called Clever, which extracts precise clinical concepts. Our results have shown that, in fact, some patient information found within the free text is missing from the structured data. This missing data can lead to incomplete and incorrect analysis. Finding and supplementing this information will provide the VA doctors a better understanding of the patients at risk.",
    	img0={"filename":"VA_rate.png", "height":"5in", "width":"8in", "caption":"Veteran Suicide Annual Report, September 2019, https://www.mentalhealth.va.gov/docs/data-sheets/2019/2019_National_Veteran_Suicide_Prevention_Annual_Report_508.pdf"})
    my_poster.add_section(title="Background",
        text="Social determinants such as homelessness, hopelessness, and social isolation are important contributors to an elevated risk of suicide. NLP of the unstructured data has succeeded in producing insights in medical applications involving such concepts in addition to major diagnoses. However, action plans were not implemented. Our methods and findings will be incorporated into our work with the VA to generate results for decision making to help improve responsiveness.")
    my_poster.add_section(title="MIMIC-III vs VA",
        text="MIMIC-III [1] is a database curated by MIT. It includes approximately 60,000 ICU admissions to a Medical Center in Boston, MA from June 2001 to October 2012. Whereas the VA database is a collection of millions of admissions from VA centers across the United States. This data is substainly more complex than what is found within MIMIC-III.",
        ref= "1. MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available at: http://www.nature.com/articles/sdata201635")
    my_poster.next_column()

    # Add sections to second column then add new column
    my_poster.add_section(title="Preliminary Work",
        text="Our LDA Topic models were trained using MIMIC free text and various Python packages known as NLTK and Gensim. These models were trained using Bag-of-Words and Term Frequency-Inverse Document Frequency (TF-IDF) on all available notes and on specific sections containing the patient’s history, which tends to have more socio-economic information that we are interested in. ")
    my_poster.add_section(title="Figures",
        #img1={"filename":"Topic_5_history.html", "height":"5.25in", "width":"10in", "caption":"Example topic when processing only the history section of discharge notes for both females and males. This topic is related to cancer, indicated by top weighted words 'cancer', 'lung' 'cell', 'mass', 'metastatic' and 'carcinoma'. For each of these words in this topic, total word count and word weight is projected."},
        img2={"filename":"Topic_9_male2.html", "height":"5.25in", "width":"10in", "caption":"Example topic when processing the history section for male patients. This topic appears to be related to the heart indicated by top weighted words 'cardiac', 'coronary' and 'aortic'. Total word count and word weight is projected."},
        img3={"filename":"SA_History_dis.html", "height":"5.25in", "width":"10in", "caption":"Frequency of dominant topics and weighted topics found within the history section of suicide attempt patient notes. Dominant topic means only the highest scoring probability is recorded for each note. Topic weightage means that all probabilities for notes are accumulated and recorded. Reference Image 5 for more information regarding these topics."})
    my_poster.next_column()

    # Add sections to third column then add new column
    # Visualize the topics and words
    
    my_poster.add_section(title="Interactive LDA Plot",
    	pyLDA={"filename":"full_lda_html.html", "height":"7in", "width":"12in", "caption": "Scroll over the various topics (1-20) to inspect the corresponding words including their frequency and weight. This LDA model was trained using only the history section of discharge notes for both male and female patients. Multidmensional Scaling (MDS) gives an estimate of similarity between the topics. Relevance metric can be adjusted to effect saliency and relevance."})
    my_poster.add_section(title="Findings", text="The LDA model trained on the history section of discharge notes was able to pull out more specific social determinants that were unrelated to their ICU stay, compared to the LDA trained on all available notes. When evaluating how the suicide attempt group, our validation group, matched to the topics produced by LDA model using all available notes, it can be seen that many of the patients experienced symptoms related to withdrawl, were scored using the Clinical Institute Withdrawal Assessment for Alcohol (CIWA), had mentions of various bodly ailments, and more. The LDA model trained using the history notes identified that the suicide attempt group had histories involving alcoholism, hypertension, heart disease, cancer and more. To narrow this down even further, the history LDA model was trained again after separating male and female patients. Female suicide attempt patients had histories of hypertension, drug abuse, depression and more, whilst male suicide attempt patients had histories of coronary artery disease, alcohol abuse, cirrhosis and hopelessness.")
    my_poster.add_section(title="Conclusion", text="NLP allows for the extraction of concepts and topics that are hidden within EHR free text data. These LDA models were successful at identifying topics related to diagnosis, procedures, routines and more. They were also successful at identifying socio-economic determinants like drug abuse. Using these results, future implementation on the CDW of the VA can identify those who are at higher risk of suicide. Current development and refining, including the use of a NLP tool called Clever on the VA database, is ongoing.")
    my_poster.add_section(title="Acknowledgments", text="This work was supported in part by the U.S. Department of Energy, Computational Research Division (CRD) of the Berkeley National Lab, and VA Million Veteran Program (MVP). Thank you again to Dr. Liu, Dr. Crivelli, Rafael, and Shirley. Thank you to Dr. Suzanne Tamang at Oak Ridge National Lab for sharing her clinical concept extractor, Clever.") 
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


