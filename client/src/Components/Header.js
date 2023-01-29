import React, { Component } from "react";
import hero from "./hero.svg";
import Fade from "react-reveal";

class Header extends Component {
  render() {
    if (!this.props.data) return null;

    const project = this.props.data.project;
    const github = this.props.data.github;
    const name = this.props.data.name;
    const description = this.props.data.description;

    return (
      <header id="home">
        <nav id="nav-wrap">
          <a className="mobile-btn" href="#nav-wrap" title="Show navigation">
            Show navigation
          </a>
          <a className="mobile-btn" href="#home" title="Hide navigation">
            Hide navigation
          </a>

          <ul id="nav" className="nav">
            <li className="current">
              <a className="smoothscroll" href="#home">
                Home
              </a>
            </li>

            <li>
              <a className="smoothscroll" href="#resume">
                Discover
              </a>
            </li>

            <li>
              <a className="smoothscroll" href="#portfolio">
                Product
              </a>
            </li>

            <li>
              <a className="smoothscroll" href="#impact">
                Impact
              </a>
            </li>

            <li>
              <a className="smoothscroll" href="#contact">
                Contact
              </a>
            </li>
          </ul>
        </nav>
        <img id="amazing" src = {hero} alt="Spirit animal in the shape of a deer standing alone in a mystical forest" />
        <div className="row banner">
          <div className="banner-text">
            <Fade bottom>
              <h1 className="responsive-headline">{"Trees Under AtTech!"}</h1>
            </Fade>
            <Fade bottom duration={1200}>
              <h3>{"Deforestation is the greatest threat to our battle against climate change."}</h3>
            </Fade>
            <hr />
            <Fade bottom duration={2000}>
              <ul className="social">
                <a href={project} className="button btn project-btn">
                  <i className="fa fa-book"></i>Learn More
                </a>
                <a href={github} className="button btn github-btn">
                  <i className="fa fa-github"></i>View Code
                </a>
              </ul>
            </Fade>
          </div>
        </div>

        <p className="scrolldown">
          <a className="smoothscroll" href="#about">
            <i className="icon-down-circle"></i>
          </a>
        </p>
      </header>
    );
  }
}

export default Header;
