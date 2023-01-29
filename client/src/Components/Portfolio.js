import React, { Component } from "react";
import Zmage from "react-zmage";
import Fade from "react-reveal";

let id = 0;
class Portfolio extends Component {
  render() {
    if (!this.props.data) return null;

    const projects = this.props.data.projects.map(function (projects) {
      let projectImage = "images/portfolio/" + projects.image;

      return (
        <div key={id++} className="columns portfolio-item">
          <div className="item-wrap">
            <Zmage alt={projects.title} src={projectImage} />
            <div style={{ textAlign: "center" }}>{projects.title}</div>
          </div>
        </div>
      );
    });

    return (
      <section id="portfolio">
        <Fade left duration={1000} distance="40px">
          <div className="row">
            <div style={{ width: "100%" }}>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"/>

              <form class="example" action="action_page.php"/>
                <input type="text" placeholder="Search.." name="search"/>
                <button id = "cool" type="submit"><i class="fa fa-search"></i></button>
              <form/>
              <iframe
                title="oihwriowherio"
                width="100%"
                height="600"
                frameborder="0"
                scrolling="no"
                marginheight="0"
                marginwidth="0"
                src="https://maps.google.com/maps?width=100%25&amp;height=600&amp;hl=en&amp;q=Link%20Rd,%20Droughill,%20Portarlington,%20Co.%20Laois,%20R32%20AE68,%20Ireland+(My%20Business%20Name)&amp;t=k&amp;z=15&amp;ie=UTF8&amp;iwloc=B&amp;output=embed"
              >
                <a href="https://www.maps.ie/distance-area-calculator.html">
                  measure acres/hectares on map
                </a>
              </iframe>
            </div>
          </div>
        </Fade>
      </section>
    );
  }
}

export default Portfolio;
