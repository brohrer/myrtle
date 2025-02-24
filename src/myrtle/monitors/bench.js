// TODO
// add ticks and labels
// Add slower, filtered reward history
//

import * as config from './config.js';
import * as art from './drawingTools.js'
import * as num from './num.js';

/*
Tweakable values
*/
// Bounds for the reward plot
const leftBorderFrac = 1 / 8;
const rightBorderFrac = 6 / 8;
const bottomBorderFrac = 5 / 8;
const topBorderFrac = 1 / 8;

const pathColor = 'CadetBlue';
const textColor = 'CadetBlue';
const lineWidthBold = 5;
const lineWidthNarrow = 1;

const textXFrac = 0.25;
const textYFrac = 0.8;

// Number of time steps of reward history to track
const historyLength = 200;

/*
Globals
*/
let msg = '';
let reward = 0.0;
let values = 0;
let step = 0;
let episode = 0;

let rewardHistory = num.zeros(historyLength);
let loopStep = num.range(-1 * historyLength, 0);

// Determines whether an update to the display is needed.
// Refreshes to true every time a non-empty message is received.
let redraw = true;

/*
Initialize the display
*/
let canvas = document.getElementById('the-canvas');
let ctx = canvas.getContext('2d');

let rewardAxes = new art.Axes(
  leftBorderFrac * canvas.width,
  rightBorderFrac * canvas.width,
  bottomBorderFrac * canvas.width,
  topBorderFrac * canvas.width,
);

let rewardUpperBound = 0.0;
let rewardLowerBound = 0.0;
rewardAxes.leftValue = num.min(loopStep);
rewardAxes.rightValue = num.max(loopStep);
rewardAxes.bottomValue = num.min(rewardHistory);
rewardAxes.topValue = num.max(rewardHistory);

let rewardChart = new art.Chart(ctx, rewardAxes);
rewardChart.color = pathColor;

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
  let maxReward = num.max(rewardHistory); 
  let minReward = num.min(rewardHistory); 
  if (rewardUpperBound < maxReward) {
    rewardUpperBound = maxReward;
    rewardAxes.topValue = rewardUpperBound;
  }
  if (rewardLowerBound > minReward) {
    rewardLowerBound = minReward;
    rewardAxes.bottomValue = rewardLowerBound;
  }

  rewardChart.render() 

  // Reward trace
  let rewardX = rewardAxes.scaleX(loopStep);
  let rewardY = rewardAxes.scaleY(rewardHistory);
  let rewardPath = new art.Path(ctx, rewardX, rewardY);
  rewardPath.lineWidth = lineWidthBold;
  rewardPath.color = pathColor;
  rewardPath.draw();

  // Reward text
  let rewardTextBody = `reward     ${reward.toFixed(3)}`;
  let rewardText = new art.Text(
    ctx,
    textXFrac * canvas.width,
    textYFrac * canvas.height,
    rewardTextBody,
  );
  rewardText.color = textColor;
  rewardText.font = "24px Courier New";
  rewardText.draw();

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
    rewardHistory.push(reward);
    rewardHistory.shift();
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
