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

    this.xAxisLabelBody = '';
    this.xAxisLabelXOffset = this.ax.width / 2;
    this.xAxisLabelYOffset = this.ax.height / 3;

    this.yAxisLabelBody = '';
    this.yAxisLabelXOffset = this.ax.width / 4;
    this.yAxisLabelYOffset = this.ax.height * 0.6;

    this.minXTicks = 4;
    this.tickXLength = this.ax.height / 20;
    this.tickXOffset = this.ax.height / 30;
    this.tickXLabelOffset = this.ax.height / 60;

    this.minYTicks = 4;
    this.tickYLength = this.ax.width / 80;
    this.tickYOffset = this.ax.width / 100;
    this.tickYLabelOffset = this.ax.width / 100;

    this.font = "14px Courier New";
    this.largeFont = "18px Courier New";
  }

  render() {
    // Baseline
    let baselineX = [this.ax.leftPixel, this.ax.rightPixel];
    let baselineY = this.ax.scaleY([this.baseline, this.baseline]);
    let baselinePath = new Path(this.ctx, baselineX, baselineY);
    baselinePath.lineWidth = this.lineWidth;
    baselinePath.color = this.color;
    baselinePath.draw();

    // x-Tick lines
    let [tickXArray, tickXLabelArray] = num.prettySpacedArray(
      this.ax.leftValue,
      this.ax.rightValue,
      this.minXTicks,
    )
    for (let i = 0; i < tickXArray.length; i++) {
      let tickXXPixels = this.ax.scaleX([tickXArray[i], tickXArray[i]]);
      let tickXYPixels = [
        this.ax.bottomPixel + this.tickXOffset,
        this.ax.bottomPixel + this.tickXLength + this.tickXOffset,
      ];
      let tickXPath = new Path(this.ctx, tickXXPixels, tickXYPixels);
      tickXPath.lineWidth = this.linewidth;
      tickXPath.color = this.color;
      tickXPath.draw();

      let tickXLabelPixel = this.ax.bottomPixel +
        this.tickXLength +
        this.tickXOffset +
        this.tickXLabelOffset;
      let tickXLabel = new Text(
        this.ctx,
        tickXXPixels[0],
        tickXLabelPixel,
        tickXLabelArray[i],
      );
      tickXLabel.color = this.color;
      tickXLabel.font = this.font;
      tickXLabel.textBaseline = 'top';
      tickXLabel.textAlign = 'center';
      tickXLabel.draw();
    }

    // y-Tick lines
    let [tickYArray, tickYLabelArray] = num.prettySpacedArray(
      this.ax.bottomValue,
      this.ax.topValue,
      this.minYTicks,
    )
    for (let i = 0; i < tickYArray.length; i++) {
      let tickYXPixels = [
        this.ax.rightPixel + this.tickYOffset,
        this.ax.rightPixel + this.tickYLength + this.tickYOffset,
      ];
      let tickYYPixels = this.ax.scaleY([tickYArray[i], tickYArray[i]]);
      let tickYPath = new Path(this.ctx, tickYXPixels, tickYYPixels);
      tickYPath.lineWidth = this.linewidth;
      tickYPath.color = this.color;
      tickYPath.draw();

      let tickYLabelPixel = this.ax.rightPixel +
        this.tickYLength +
        this.tickYOffset +
        this.tickYLabelOffset;
      let tickYLabel = new Text(
        this.ctx,
        tickYLabelPixel,
        tickYYPixels[0],
        tickYLabelArray[i],
      );
      tickYLabel.color = this.color,
      tickYLabel.font = this.font,
      tickYLabel.textBaseline = 'middle';
      tickYLabel.textAlign = 'left';
      tickYLabel.draw();
    }

    // x-axis label
    let xAxisLabelXPixel = this.ax.leftPixel + this.xAxisLabelXOffset;
    let xAxisLabelYPixel = this.ax.bottomPixel + this.xAxisLabelYOffset;
    let xAxisLabel = new Text(
      this.ctx,
      xAxisLabelXPixel,
      xAxisLabelYPixel,
      this.xAxisLabelBody,
    );
    xAxisLabel.color = this.color,
    xAxisLabel.font = this.largeFont,
    xAxisLabel.textBaseline = 'top';
    xAxisLabel.textAlign = 'center';
    xAxisLabel.draw();

    // y-axis label
    let yAxisLabelXPixel = this.ax.rightPixel + this.yAxisLabelXOffset;
    let yAxisLabelYPixel = this.ax.bottomPixel - this.yAxisLabelYOffset;
    let yAxisLabel = new Text(
      this.ctx,
      yAxisLabelXPixel,
      yAxisLabelYPixel,
      this.yAxisLabelBody,
    );
    yAxisLabel.color = this.color,
    yAxisLabel.font = this.largeFont,
    yAxisLabel.textBaseline = 'left';
    yAxisLabel.textAlign = 'middle';
    yAxisLabel.draw();

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
