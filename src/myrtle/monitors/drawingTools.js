import * as num from './num.js';

// Canvas API reference
// https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D
// choose from colors
// https://www.w3schools.com/colors/colors_names.asp

export class Axes {
  // Keep in mind that pixel 0 is at the top of the canvas
  constructor(
    leftPixel,
    rightPixel,
    bottomPixel,
    topPixel,
    leftValue=0.0,
    rightValue=1.0,
    bottomValue=0.0,
    topValue=1.0,
  ) {
    this.leftPixel = leftPixel;
    this.rightPixel = rightPixel;
    this.bottomPixel = bottomPixel;
    this.topPixel = topPixel;
    this.leftValue = leftValue;
    this.rightValue = rightValue;
    this.bottomValue = bottomValue;
    this.topValue = topValue;

    this.width = rightPixel - leftPixel;
    this.height = bottomPixel - topPixel;
  }

  // Convert an array of x-values to horizontal pixel positions
  scaleX(xArray) {
    return this.scaleArray(
      xArray,
      this.leftValue,
      this.rightValue,
      this.leftPixel,
      this.rightPixel,
    )
  }

  // Convert an array of y-values to vertical pixel positions
  scaleY(yArray) {
    return this.scaleArray(
      yArray,
      this.bottomValue,
      this.topValue,
      this.bottomPixel,
      this.topPixel,
    )
  }

  // Convert an array of values to pixel positions
  scaleArray(valueArray, minValue, maxValue, minPixel, maxPixel) {
    let pixels = [];
    for (let i = 0; i < valueArray.length; i++) {
      pixels[i] = this.scaleValue(
        valueArray[i],
        minValue,
        maxValue,
        minPixel,
        maxPixel,
      );
    }
    return pixels;
  }

  // Convert a value to a pixel position.
  scaleValue(
      value,
      minValue,
      maxValue,
      minPixel,
      maxPixel,
  ) {
    let valueScaled = (value - minValue) / (maxValue - minValue);
    return Math.round(minPixel + valueScaled * (maxPixel - minPixel));
  }
}

export class Path {
  constructor(canvasContext, x, y) {
    this.ctx = canvasContext;
    this.x = x;
    this.y = y;
    this.color = 'black';
    this.linewidth = 1;
    this.lineJoin = "round"; 
  }

  draw() {
    this.ctx.lineWidth = this.lineWidth;
    this.ctx.lineJoin = this.lineJoin;
    this.ctx.strokeStyle = this.color;

    this.ctx.beginPath();
    this.ctx.moveTo(this.x[0], this.y[0]);
    for (let i = 1; i < this.y.length; i++) {
      this.ctx.lineTo(this.x[i], this.y[i]);
    }
    this.ctx.stroke();
  }
}

// All the decorations and accoutrements of a 2 dimensional plot,
// like baselines and ticks. Opinionated, non-standard design.
export class Chart {
  constructor(canvasContext, axes) {
    this.ctx = canvasContext;
    this.ax = axes;
    this.backgroundColor = 'Black';
    this.color = 'White';
    this.lineWidth = 1.0;
    this.baseline = 0.0;
    this.minTicks = 4;
    this.tickLength = this.ax.width / 40;
    this.tickOffset = this.ax.width / 40;
    this.tickLabelOffset = this.ax.width / 40;
    this.font = "14px Courier New";
  }

  render() {
    // Baseline
    let baselineX = [this.ax.leftPixel, this.ax.rightPixel];
    let baselineY = this.ax.scaleY([this.baseline, this.baseline]);
    let baselinePath = new Path(this.ctx, baselineX, baselineY);
    baselinePath.lineWidth = this.lineWidth;
    baselinePath.color = this.color;
    baselinePath.draw();

    // Tick lines
    let [tickYArray, tickYLabelArray] = num.prettySpacedArray(
      this.ax.bottomValue,
      this.ax.topValue,
      this.minTicks,
    )
    for (let i = 0; i < tickYArray.length; i++) {
      let tickX = [
        this.ax.rightPixel + this.tickOffset,
        this.ax.rightPixel + this.tickLength + this.tickOffset,
      ];
      let tickYPixel = this.ax.scaleY([tickYArray[i], tickYArray[i]]);
      let tickPath = new Path(this.ctx, tickX, tickYPixel);
      tickPath.lineWidth = this.linewidth;
      tickPath.color = this.color;
      tickPath.draw();

      let tickLabelX = this.ax.rightPixel +
        this.tickLength +
        this.tickOffset +
        this.tickLabelOffset;
      let tickLabel = new Text(
        this.ctx,
        tickLabelX,
        tickYPixel[0],
        tickYLabelArray[i],
      );
      tickLabel.color = this.color,
      tickLabel.font = this.font,
      tickLabel.draw();
    }
  }
}

export class Text {
  constructor(canvasContext, x, y, text) {
    this.ctx = canvasContext;
    this.x = x;
    this.y = y;
    this.text = text;
    this.color = "black";
    this.font = "24px Courier New";
    this.textAlign = "left";
    this.textBaseline = "middle";
  }
 
  draw () {
    this.ctx.fillStyle = this.color;
    this.ctx.font = this.font;
    this.ctx.textAlign = this.textAlign;
    this.ctx.textBaseline = this.textBaseline;
    this.ctx.fillText(this.text, this.x, this.y);
  }
}
