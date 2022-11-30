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

def generate_navbar(page):
    return dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink('By Salary', href='/salary' )),
        html.Div(className="divider"),
        dbc.NavItem(dbc.NavLink('By Field', href='/field')),
        html.Div(className="divider"),
        dbc.NavItem(dbc.NavLink('Prediction', href='/prediction')),
    ],
    brand="StubEnhancer",
    brand_href="/",
    color="#27293D",
    class_name="header",
    dark="true"
)