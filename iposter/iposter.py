import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

#--
def PosterSection(title, color, children=[], height=None):
    style = {"padding-bottom":"15px"}
    if height: style["height"] = height
    layout = dbc.Row(
    dbc.Card([
        dbc.Card([
            html.H4(title,style={"color":"white","text-align":"center","font-size":"66px", "font-family": "Arial", "font-weight":"bold"}),
        ],style={"background":color,"padding":"20px"}),
        dbc.Card(dbc.Container(children,fluid=True),style={"padding":"20px"})],
        body=True),style=style)
    return layout

#--
def PosterColumn(sections):
    layout = dbc.Col(
    dbc.Card(
    sections,
    body=True,
    style={"padding":"45px"}))
    return layout

#--
def Header(title, authors, institutions, logo, banner_color):
    layout = dbc.Card(
    dbc.Row(
    [dbc.Col(html.Img(src="SHI.png", style={'height':'2in', "width":"5in"}),style={"display":"flex","justify-content":"flex-end","padding-left":"80px"},width=1.5),
     dbc.Col([dbc.Row(html.Img(src="logo1.png", style={'height':'1.75in', "width":"5in","padding-left":"50px"})),
              dbc.Row(html.Img(src="logo2.png", style={'height':'1.75in', "width":"5in", "padding-left":"50px"}))],width=1.5),
     dbc.Col([
        dbc.Row(dbc.Container(title,fluid=True)),
        dbc.Row(dbc.Container(authors,fluid=True)),
        dbc.Row(dbc.Container(institutions,fluid=True)),
        ],style={"padding-top":"15px"}),
     dbc.Col(html.Img(src="qrcode1.png", style={'height':'4in', "width":"4in"}),style={"display":"flex","justify-content":"flex-end","padding-right":"80px"},width=1.5)],
    style={'height':'4in', "width":"42in"},justify="center"),
    style={"background": banner_color},
    body=True)
    return layout

#-
def Poster(title, authors, institutions, logo, columns, bg_color, banner_color):
    layout = html.Div(
    dbc.Col([
        dbc.Row(Header(title, authors, institutions, logo, banner_color), style={"padding":"15px"}),
        dbc.DropdownMenuItem(divider=True),
        dbc.Row(columns)], style={"height": "32in"}),
    style={"background": bg_color, "height": "36in", "width": "42in"})

    return layout

#-
class iPoster:

    #--
    def __init__(self, title, authors_dict, logo, banner_color="#0033cc", text_color="white"):
        self.poster_title = title
        self.authors = authors_dict
        self.logo= logo
        self.bg_color = "#e6eeff"
        self.banner_color = banner_color
        self.text_color = text_color
        self.sects = []
        self.cols = []
        self.figure_counter = 0

    #--
    def _header_components(self):
    	index_dict = dict([(x[1], x[0]+1) for x in enumerate(set(reversed(sum(list(self.authors.values()), []))))])
    	authors = []
    	for a in self.authors:
        	authors += [a]
        	authors += [html.Sup(",".join([str(index_dict[x]) for x in self.authors[a]]))]
        	authors += [", "]
    	authors = authors[:-1]
    	insts = []
    	for s in index_dict:
        	insts += [html.Sup(index_dict[s])]
        	insts += [s]
        	insts += [", "]
    	insts = insts[:-1]
    	title = html.H1(self.poster_title, style={"text-align":"center","font-size":"89px","color":self.text_color, "font-family": "Arial", "font-weight":"bold"})
    	authors = html.H2(authors,style={"text-align":"center","font-size":"59px","color":self.text_color, "font-family": "Arial", "font-weight":"bold"})
    	institutions = html.H3(insts, style={"text-align":"center","font-size":"48px","color":self.text_color, "font-family": "Arial", "font-weight":"bold"})
    	return title, authors, institutions

    #--
    def add_section(self, title, text=None, ref=None, img0=None, img1=None, img2=None, img3=None, img4=None, plot=None, pyLDA=None, color="#2f608b", height=None, children=[]):
        childs = []
        if text: 
        	childs.append(html.P(text, style={"font-family": "Georgia", "font-size":"28px"}))
        if ref: 
        	childs.append(html.P(ref, style={"font-family": "Georgia", "font-size":"24px"}))
        if img0:
            childs.append(html.Img(src=img0["filename"], style={"height":img0["height"], "width":img0["width"], "margin-left": "auto", "margin-right": "auto", "display": "block"}))
            self.figure_counter += 1
            childs.append(html.P("Figure {}. ".format(self.figure_counter) + img0["caption"], style={"font-family": "Georgia","font-size":"24px", "font-weight":"normal", "text-align":"center"}))
        if img1:
            childs.append(html.Iframe(src=img1["filename"], style={"height":img1["height"], "width":img1["width"], "margin-left": "auto", "margin-right": "auto", "display": "block"}))
            self.figure_counter += 1
            childs.append(html.P("Figure {}. ".format(self.figure_counter) + img1["caption"], style={"font-family": "Georgia","font-size":"24px", "font-weight":"normal", "text-align":"center"}))
        if img2:
            childs.append(html.Iframe(src=img2["filename"], style={"height":img2["height"], "width":img2["width"], "margin-left": "auto", "margin-right": "auto", "display": "block"}))
            self.figure_counter += 1
            childs.append(html.P("Figure {}. ".format(self.figure_counter) + img2["caption"], style={"font-family": "Georgia", "font-size":"24px", "font-weight":"normal", "text-align":"center"}))
        if pyLDA:
            childs.append(html.Iframe(src=pyLDA["filename"], style={"height":pyLDA["height"], "width":pyLDA["width"]}))
            self.figure_counter += 1
            childs.append(html.P("Figure {}. ".format(self.figure_counter) + pyLDA["caption"], style={"font-family": "Georgia", "font-size":"24px", "font-weight":"normal", "text-align":"center"}))
        if img3:
            childs.append(html.Img(src=img3["filename"], style={"height":img3["height"], "width":img3["width"], "margin-left": "auto", "margin-right": "auto", "display": "block"}))
            self.figure_counter += 1
            childs.append(html.P("Figure {}. ".format(self.figure_counter) + img3["caption"], style={"font-family": "Georgia", "font-size":"24px", "font-weight":"normal", "text-align":"center"}))
        if img4:
            childs.append(html.Img(src=img4["filename"], style={"height":img4["height"], "width":img4["width"], "margin-left": "auto", "margin-right": "auto", "display": "block"}))
            self.figure_counter += 1
            childs.append(html.P("Figure {}. ".format(self.figure_counter) + img4["caption"], style={"font-family": "Georgia", "font-size":"24px", "font-weight":"normal", "text-align":"center"}))
        if plot:
            childs.append(dcc.Graph(figure=plot["fig"]))
            self.figure_counter += 1
            childs.append(html.P("Figure {}. ".format(self.figure_counter) + plot["caption"], style={"font-family": "Georgia", "font-size":"24px", "font-weight":"normal"}))

        childs += children
        self.sects.append(PosterSection(title, color, childs, height=height))

    #--
    def next_column(self):
        if len(self.sects) > 0:
            self.cols.append(PosterColumn(self.sects))
            self.sects = []

    #--
    def compile(self):
        title, authors, institutions = self._header_components()
        return Poster(title, authors, institutions, self.logo, self.cols, self.bg_color, self.banner_color)
