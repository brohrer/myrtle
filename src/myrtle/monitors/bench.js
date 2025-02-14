import { arrayMin, arrayMax } from './num.js';
import { mq_host, mq_port } from './config.js';

const addr = `ws://${mq_host}:${mq_port}`;
const socket = new WebSocket(addr);

// get canvas and context
let canvas = document.getElementById('the-canvas'),
ctx = canvas.getContext('2d');

// parameters for the display
const leftBorder = canvas.width / 8
const rightBorder = canvas.width * 6 / 8
const topBorder = canvas.width / 8
const bottomBorder = canvas.width * 5 / 8

let msg = '';
let reward = 0.0;
let values = 0;
let step = 0;
let episode = 0;

const historyLength = 1000;
var rewardHistory = new Array(historyLength).fill(0);
let x = new Array(historyLength).fill(0);
let y = new Array(historyLength).fill(0);

socket.onmessage = (event) => {
  console.log(event.data);
  let obj = JSON.parse(event.data);
  if (obj.message !== ""){
    msg = obj.message;
    values = JSON.parse(msg);
    step = values.loop_step;
    episode = values.episode;
    // TODO: sum all rewards
    reward = values.rewards[0];
    rewardHistory.push(reward);
    rewardHistory.shift();
  };
};

// Main APP loop
function loop()
{
  try {
    socket.send('{"action": "get", "topic": "world_step"}');
  }
  catch(InvalidStateError) {
    console.log("InvalidStateError caught");
  }

  // draw
  // choose from colors
  // https://www.w3schools.com/colors/colors_names.asp
  ctx.fillStyle = 'Black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'CadetBlue';

  calculateY()
  ctx.beginPath();
  ctx.moveTo(x[0], y[0]);
  for (let i = 1; i < y.length; i++) {
    ctx.lineTo(x[i], y[i]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "CadetBlue";
  }
  ctx.stroke();

  let y_text = canvas.height * 0.8,
  x_text = canvas.width / 4;
  ctx.font = "24px Courier New";
  ctx.fillText(`reward     ${reward.toFixed(3)}`, x_text, y_text);

  //request next frame
  requestAnimationFrame(loop);
};

calculateX()
loop()

function calculateX()
{
  for (let i = 0; i < x.length; i++) {
    x[i] = leftBorder + (i / (x.length - 1)) * (rightBorder - leftBorder);
  }
}

function calculateY()
{
  let min_reward = arrayMin(rewardHistory);
  let max_reward = arrayMax(rewardHistory);
  for (let i = 0; i < y.length; i++) {
    let reward_norm = (rewardHistory[i] - min_reward) / (max_reward - min_reward);
    y[i] = bottomBorder + reward_norm * (topBorder - bottomBorder) 
  }
}
/*
function getURLParameter(sParam)
{
  let sPageURL = window.location.search.substring(1);
  let sURLVariables = sPageURL.split('&');
  for (let i = 0; i < sURLVariables.length; i++)
  {
    let sParameterName = sURLVariables[i].split('=');
    if (sParameterName[0] == sParam)
    {
      return sParameterName[1];
    }
  }
}
*/
