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
        text="",
    	img0={"filename":"VA_rate.png", "height":"5in", "width":"8in", "caption":"Veteran Suicide Annual Report, September 2019, https://www.mentalhealth.va.gov/docs/data-sheets/2019/2019_National_Veteran_Suicide_Prevention_Annual_Report_508.pdf"})
    my_poster.add_section(title="Background",
        text="Social determinants such as homelessness, hopelessness, and social isolation are important contributors to an elevated risk of suicide. NLP of the unstructured data has succeeded in producing insights in medical applications involving such concepts in addition to major diagnoses. However, action plans were not implemented. Our methods and findings will be incorporated into our work with the VA to generate results for decision making to help improve responsiveness.")
    my_poster.add_section(title="MIMIC-III vs VA",
        text="MIMIC-III [1] is a database curated by MIT. It includes approximately 60,000 ICU admissions to a Medical Center in Boston, MA from June 2001 to October 2012. Whereas the VA database is a collection of millions of admissions from VA centers across the United States. This data is substantially more complex than what is found within MIMIC-III.",
        ref= "1. MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available at: http://www.nature.com/articles/sdata201635")
    my_poster.add_section(title="Preliminary Work",
        text="Our LDA Topic models were trained using MIMIC free text and various Python packages known as NLTK and Gensim. These models were trained using Bag-of-Words and Term Frequency-Inverse Document Frequency (TF-IDF) on all available notes and on specific sections containing the patient’s history. ")
    my_poster.next_column()

    # Add sections to second column then add new column
    my_poster.add_section(title="Interactive Figures",
        #img1={"filename":"Topic_5_history.html", "height":"5.25in", "width":"10in", "caption":"Example topic when processing only the history section of discharge notes for both females and males. This topic is related to cancer, indicated by top weighted words 'cancer', 'lung' 'cell', 'mass', 'metastatic' and 'carcinoma'. For each of these words in this topic, total word count and word weight is projected."},
        img2={"filename":"Topic_9_male2.html", "height":"5.25in", "width":"10in", "caption":"Example topic when processing the history section for male patients. This topic appears to be related to the heart indicated by top weighted words 'cardiac', 'coronary' and 'aortic'. Total word count and word weight is projected."},
        img3={"filename":"SA_History_dis.html", "height":"5.25in", "width":"10in", "caption":"Frequency of dominant topics and weighted topics found within the history section of both male and female suicide attempt patient notes. Dominant topic means only the highest scoring probability is recorded for each note. Topic weightage means that all probabilities for notes are accumulated and recorded. Reference Figure 3 for more information regarding these topics."})
    	pyLDA={"filename":"full_lda_html.html", "height":"7in", "width":"12in", "caption": "Figure 2 LDA topics and their corresponding word frequency and weight. Multidmensional Scaling (MDS) gives an estimate of similarity between the topics. Relevance metric can be adjusted to effect saliency and relevance."})
    my_poster.add_section(title="LDA Findings", text="The LDA model trained on the history section of notes was able to pull out more specific social determinants that were unrelated to their ICU stay, compared to the LDA trained on all available notes. These include histories involving alcoholism, hypertension, heart disease, cancer and more. To narrow this down even further, female patients had histories of hypertension, drug abuse and depression, whilst male patients had histories of coronary artery disease, alcohol abuse, cirrhosis and hopelessness.")
    my_poster.next_column()

    # Add sections to third column then add new column
    # Visualize the topics and words
    my_poster.add_section(title="Current Work",
        text="Clever") 
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


