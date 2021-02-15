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
                        authors_dict={"Destinee Morrow" : ["Lawrence Berkeley National Laboratory"], # Authors in {student, mentors, PI} order
                                      "Rafael Zamora-Resendiz" : ["Lawrence Berkeley National Laboratory"],
                                      "Xiange Wang" : ["Lawrence Berkeley National Laboratory"],
                                      "Xinlian Liu" : ["Hood College", "Lawrence Berkeley National Laboratory"],
                                      "Silvia Crivelli" : ["Lawrence Berkeley National Laboratory"]},
                        logo = "SHI.png", # Home institution logo
                        banner_color=colors.DOE_GREEN, # Color of banner header; colors has preset colors
                        text_color=colors.WHITE)

    # Add sections to first column then add new column
    my_poster.add_section(title="Abstract",
        text="Suicide is a major national health concern. Current theories in mental health studies have failed to create predictive models that can be used in suicide prevention. Adoption of Electronic Health Records (EHR) has led to a sharp increase in clinical data in forms of structured data, such as demographic information, medications, billing codes, etc, and unstructured data such as physician notes. Processing, formatting, and parsing of such notes are essential in helping produce clinical decisions. Preliminary work showed that our Latent Dirichlet Allocation (LDA) models are sensitive enough to pick up different topics related to determinants of suicide between genders. Currently, we’ve been working with doctors at the U.S Department of Veterans Affairs (VA) and using a secondary natural language processing (NLP) tool called Clincal Event Recognizer (CLEVER). Our results have shown that important patient information found within the free text is missing from the structured data. Finding and supplementing this information will provide the VA doctors a better understanding of the patients at risk.",
    	img0={"filename":"death_causes.png", "height":"5.5in", "width":"10in", "caption":" CDC’s National Center for Health Statistics, https://www.nimh.nih.gov/health/statistics/suicide.shtml"})
    my_poster.add_section(title="Background",
        text="Social determinants such as homelessness, hopelessness, and social isolation are important contributors to an elevated risk of suicide. NLP of the unstructured text has succeeded in producing insights in medical applications involving such concepts in addition to major diagnoses. However, action plans were not implemented. Our methods and findings will generate results for VA decision making to help improve responsiveness.")
    my_poster.add_section(title="MIMIC-III vs VA",
        text="MIMIC-III [1] is a database curated by MIT. It includes approximately 60,000 ICU admissions to a Medical Center in Boston, MA from June 2001 to October 2012. The VA database is a collection of millions of admissions from VA centers across the United States. VA data is substantially more complex than what is found within MIMIC-III.",
        ref= "1. MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available at: http://www.nature.com/articles/sdata201635")
    my_poster.add_section(title="Preliminary Work",
        text="Latent Dirichlet Allocation (LDA) is a statistical model that finds topics in free format text by using word count and frequency. Our LDA Topic models were trained using MIMIC-III discharge notes and various Python packages known as NLTK and Gensim, which include the use of Bag-of-Words and Term Frequency-Inverse Document Frequency (TF-IDF).")
    my_poster.next_column()

    # Add sections to second column then add new column
    my_poster.add_section(title="Interactive Figures",
        #img1={"filename":"Topic_5_history.html", "height":"5.25in", "width":"10in", "caption":"Example topic when processing only the history section of discharge notes for both females and males. This topic is related to cancer, indicated by top weighted words 'cancer', 'lung' 'cell', 'mass', 'metastatic' and 'carcinoma'. For each of these words in this topic, total word count and word weight is projected."},
        img1={"filename":"Topic_9_male2.html", "height":"5.25in", "width":"10in", "caption":"LDA example when processing the history section of male patient notes. This topic appears to be related to the heart indicated by top weighted words 'cardiac', 'coronary' and 'aortic'. All topic concepts are assumptions based on LDA output, word weight and frequency."},
        img2={"filename":"SA_History_dis.html", "height":"5.25in", "width":"10in", "caption":"Frequency of LDA dominant topics and weighted topics found within the history section of both male and female suicide attempt patient notes. Dominant topic means only the highest scoring probability for each note is recorded. Topic weightage means that all probabilities for notes are accumulative. This model was trained to output 20 potential topics."},
        pyLDA={"filename":"full_lda_html.html", "height":"7.5in", "width":"12in", "caption": "Figure 3 LDA topics and their corresponding word frequency and weight. Multidmensional Scaling (MDS) gives an estimate of similarity between the topics. Relevance metric can be adjusted to effect saliency and relevance."})
    my_poster.add_section(title="LDA Findings", text="The LDA models trained on the history section of notes were able to pull out more specific social determinants that were unrelated to their ICU stay, compared to the LDA trained on all available notes. These include histories involving alcoholism, hypertension, heart disease, cancer and more. To narrow this down even further, female patients had histories of hypertension, drug abuse and depression, whilst male patients had histories of coronary artery disease, alcohol abuse, cirrhosis and hopelessness.")
    my_poster.next_column()

    # Add sections to third column then add new column
    # Visualize the topics and words
    my_poster.add_section(title="Current Work",
        text="CLEVER is an adaptive clinical concept extractor. This method does not require any machine learning training and is broken down into the three steps seen below. Modification of the lexicon can include any and all desired clinical concepts for precise output.",
        img3={"filename":"clever_flow.png", "height":"6.75in", "width":"10in", "caption":"Flowchart of CLEVER."}) 
    my_poster.add_section(title="Clever Output",
        img4={"filename":"clever_output.png", "height":"4.50in", "width":"10in", "caption":"Example output of CLEVER that shows positive indication of homelessness found within the free text. Personal Identification Information has been removed."})
    my_poster.add_section(title="Conclusion", text="NLP allows for the extraction of concepts and topics that are hidden within EHR free text. These LDA models were successful at identifying topics related to diagnosis, procedures, routines and more. They also identifed socio-economic determinants like drug abuse. As a follow-up, CLEVER successfully identified the unique patients with these determinants. These results can then be used to identify potentially missing structured data and those who are at higher risk of suicide. Current development and refining is ongoing.")
    my_poster.add_section(title="Acknowledgments", text="This work was supported in part by the U.S. Department of Energy, Computational Research Division (CRD) of the Berkeley National Lab, and VA Million Veteran Program (MVP). Thank you to Sustainable Horizons Institute and Sustainable Research Pathways for making this conference attendence possible. Thank you to Dr. Suzanne Tamang at Oak Ridge National Lab for sharing her clinical concept extractor, CLEVER.") 
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


