import * as config from './config.js';
import * as art from './drawingTools.js'
import * as num from './num.js';

/*
Tweakable values
*/
// Bounds for the reward plots
const fastLeftBorderFrac = 0.125;
const fastRightBorderFrac = 0.75;
const fastBottomBorderFrac = 0.28;
const fastTopBorderFrac = 0.13;

const medLeftBorderFrac = 0.125;
const medRightBorderFrac = 0.75;
const medBottomBorderFrac = 0.55;
const medTopBorderFrac = 0.40;

const slowLeftBorderFrac = 0.125;
const slowRightBorderFrac = 0.75;
const slowBottomBorderFrac = 0.83;
const slowTopBorderFrac = 0.68;

const pathColor = 'CadetBlue';
const textColor = 'CadetBlue';
const lineWidthBold = 3;
const lineWidthNarrow = 1;

const xLabel = 'time steps';
const yLabel = 'reward';

const minXTicks = 5;
const minYTicks = 3;

// Number of time steps of reward history to track
const fastHistoryLength = 200;
const medHistoryLength = 200;
const medHistoryBinSize = 20;
const slowHistoryLength = 200;
const slowHistoryBinSize = 500;

/*
Globals
*/
let msg = '';
let reward = 0.0;
let totalMedReward = 0;
let totalSlowReward = 0;
let values = 0;
let step = 0;
let episode = 0;

let fastRewardHistory = num.zeros(fastHistoryLength);
let fastLoopStep = num.range(-1 * fastHistoryLength, 0);

let medRewardHistory = num.zeros(medHistoryLength);
let medLoopStep = num.intervalSpacedArray(
  -1 * medHistoryLength * medHistoryBinSize, 0, medHistoryBinSize, true);

let slowRewardHistory = num.zeros(slowHistoryLength);
let slowLoopStep = num.intervalSpacedArray(
  -1 * slowHistoryLength * slowHistoryBinSize, 0, slowHistoryBinSize, true);

// Determines whether an update to the display is needed.
// Refreshes to true every time a non-empty message is received.
let redraw = true;

/*
Initialize the display
*/
let canvas = document.getElementById('the-canvas');
let ctx = canvas.getContext('2d');

let fastAxes = new art.Axes(
  fastLeftBorderFrac * canvas.width,
  fastRightBorderFrac * canvas.width,
  fastBottomBorderFrac * canvas.height,
  fastTopBorderFrac * canvas.height,
);

let fastUpperBound = 0.0;
let fastLowerBound = 0.0;
fastAxes.leftValue = num.min(fastLoopStep);
fastAxes.rightValue = num.max(fastLoopStep);
fastAxes.bottomValue = num.min(fastRewardHistory);
fastAxes.topValue = num.max(fastRewardHistory);

let fastChart = new art.Chart(ctx, fastAxes);
fastChart.color = pathColor;
fastChart.minXTicks = minXTicks;
fastChart.minYTicks = minYTicks;
// fastChart.xAxisLabelBody = xLabel;
// fastChart.yAxisLabelBody = yLabel;

let medAxes = new art.Axes(
  medLeftBorderFrac * canvas.width,
  medRightBorderFrac * canvas.width,
  medBottomBorderFrac * canvas.height,
  medTopBorderFrac * canvas.height,
);

let medUpperBound = 0.0;
let medLowerBound = 0.0;
medAxes.leftValue = num.min(medLoopStep);
medAxes.rightValue = num.max(medLoopStep);
medAxes.bottomValue = num.min(medRewardHistory);
medAxes.topValue = num.max(medRewardHistory);

let medChart = new art.Chart(ctx, medAxes);
medChart.color = pathColor;
medChart.minXTicks = minXTicks;
medChart.minYTicks = minYTicks;
// medChart.xAxisLabelBody = xLabel;
medChart.yAxisLabelBody = yLabel;


let slowAxes = new art.Axes(
  slowLeftBorderFrac * canvas.width,
  slowRightBorderFrac * canvas.width,
  slowBottomBorderFrac * canvas.height,
  slowTopBorderFrac * canvas.height,
);

let slowUpperBound = 0.0;
let slowLowerBound = 0.0;
slowAxes.leftValue = num.min(slowLoopStep);
slowAxes.rightValue = num.max(slowLoopStep);
slowAxes.bottomValue = num.min(slowRewardHistory);
slowAxes.topValue = num.max(slowRewardHistory);

let slowChart = new art.Chart(ctx, slowAxes);
slowChart.color = pathColor;
slowChart.minXTicks = minXTicks;
slowChart.minYTicks = minYTicks;
slowChart.xAxisLabelBody = xLabel;
// slowChart.yAxisLabelBody = yLabel;

/*    
Establish connection and handle communication
*/
const addr = `ws://${config.mq_host}:${config.mq_port}`;
const socket = new WebSocket(addr);

// The main animation loop
loop();

/*
Draw the display
*/
function render() {
  // Background
  let backgroundColor = 'Black';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = backgroundColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Update the limits of the axes
  let maxReward = num.max(fastRewardHistory); 
  let minReward = num.min(fastRewardHistory); 
  if (fastUpperBound < maxReward) {
    fastUpperBound = maxReward;
    fastAxes.topValue = fastUpperBound;

    medUpperBound = fastUpperBound;
    medAxes.topValue = medUpperBound;

    slowUpperBound = fastUpperBound;
    slowAxes.topValue = slowUpperBound;
  }
  if (fastLowerBound > minReward) {
    fastLowerBound = minReward;
    fastAxes.bottomValue = fastLowerBound;

    medLowerBound = fastLowerBound;
    medAxes.bottomValue = medLowerBound;

    slowLowerBound = fastLowerBound;
    slowAxes.bottomValue = slowLowerBound;
  }

  fastChart.render() 
  medChart.render() 
  slowChart.render() 

  // Reward trace
  let fastRewardX = fastAxes.scaleX(fastLoopStep);
  let fastRewardY = fastAxes.scaleY(fastRewardHistory);
  let fastRewardPath = new art.Path(ctx, fastRewardX, fastRewardY);
  fastRewardPath.lineWidth = lineWidthBold;
  fastRewardPath.color = pathColor;
  fastRewardPath.draw();

  let medRewardX = medAxes.scaleX(medLoopStep);
  let medRewardY = medAxes.scaleY(medRewardHistory);
  let medRewardPath = new art.Path(ctx, medRewardX, medRewardY);
  medRewardPath.lineWidth = lineWidthBold;
  medRewardPath.color = pathColor;
  medRewardPath.draw();

  let slowRewardX = slowAxes.scaleX(slowLoopStep);
  let slowRewardY = slowAxes.scaleY(slowRewardHistory);
  let slowRewardPath = new art.Path(ctx, slowRewardX, slowRewardY);
  slowRewardPath.lineWidth = lineWidthBold;
  slowRewardPath.color = pathColor;
  slowRewardPath.draw();

  redraw = false;
}

// Handle communication and messages
socket.onmessage = (event) => {
  let obj = JSON.parse(event.data);
  if (obj.message !== ""){
    redraw = true;
    msg = obj.message;
    values = JSON.parse(msg);
    step = values.loop_step;
    episode = values.episode;

    // sum all rewards
    reward = 0.0;
    for (let i = 0; i < values.rewards.length; i++) {
      try {
        reward += values.rewards[i];
      } catch(err) {
        // Handle the case where reward is absent
        console.log(err);
      }
    }
    fastRewardHistory.push(reward);
    fastRewardHistory.shift();

    totalMedReward += reward;
    totalSlowReward += reward;

    if (step % medHistoryBinSize == 0) {
      medRewardHistory.push(totalMedReward / medHistoryBinSize);
      medRewardHistory.shift();
      totalMedReward = 0;
    }

    if (step % slowHistoryBinSize == 0) {
      slowRewardHistory.push(totalSlowReward / slowHistoryBinSize);
      slowRewardHistory.shift();
      totalSlowReward = 0;
    }
  };
};

function loop() {
  try {
    socket.send('{"action": "get", "topic": "world_step"}');
  }
  catch(InvalidStateError) {
    console.log("InvalidStateError caught");
  }

  if (redraw) {
    render();
  }

  //request next frame
  requestAnimationFrame(loop);
};
