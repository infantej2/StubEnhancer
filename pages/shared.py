import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

def generate_header_button(name, href, current_page):
    is_current_page = href[1:].lower() == current_page.split('.')[-1].lower()

    if is_current_page:
        return dcc.Link(
            html.Button(
                name,
                className="button-disabled",
                disabled=True
            ),
            href=href,
            refresh=True
        )
    else:
        return dcc.Link(
            html.Button(
                name,
                className="button"
            ),
            href=href,
            refresh=True
        )

def generate_header(page):
    return html.Header(
        children=[
            #generate_header_button('Home', '/', page),
            generate_header_button('By Salary', '/salary', page),
            html.Div(className="divider"),
            generate_header_button('By Field', '/field', page),
            html.Div(className="divider"),
            generate_header_button('Prediction', '/prediction', page),
        ],
        className="header",
        style={"textAlign":"center", "margin":"10px"}
    )

def generate_navbar_item(text, href, current_page):
    is_current_page = href[1:].lower() == current_page.split('.')[-1].lower()

    if is_current_page:
        return dbc.NavItem(dbc.NavLink(text, href=href, style={"color": "#D84FD2"}))
    else:
        return dbc.NavItem(dbc.NavLink(text, href=href))

def generate_navbar(page):
    return dbc.NavbarSimple(
    children=[
        generate_navbar_item('By Salary', '/salary', page),
        html.Div(className="divider"),
        generate_navbar_item('By Field', '/field', page),
        html.Div(className="divider"),
        generate_navbar_item('Prediction', '/prediction', page),
    ],
    brand="StubEnhancer",
    brand_href="/",
    color="#27293D",
    class_name="header",
    id="header",
    dark="true"
)