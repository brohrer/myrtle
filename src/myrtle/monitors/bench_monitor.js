let host = getURLParameter('host');
let port = getURLParameter('port');

const addr = `ws://${host}:${port}`;
const socket = new WebSocket(addr);

// get canvas and context
let canvas = document.getElementById('the-canvas'),
ctx = canvas.getContext('2d');

// parameters for the display
const leftBorder = canvas.width / 8
const rightBorder = canvas.width * 7 / 8
const topBorder = canvas.width / 8
const bottomBorder = canvas.width * 5 / 8

let msg = '';
let reward = 0.0;
let step = 0;
let episode = 0;

const historyLength = 1000;
var rewardHistory = new Array(historyLength).fill(0);
let x = new Array(historyLength).fill(0);
let y = new Array(historyLength).fill(0);

//socket.onopen = () => {
//  console.log('ws opened on browser');
//  socket.send('hello world');
//};

socket.onmessage = (event) => {
  console.log(event.data);
  obj = JSON.parse(event.data);
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
  event.data = reward;
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
    ctx.lineWidth = 1;
    ctx.strokeStyle = "CadetBlue";
  }
  ctx.stroke();

  let y_text = canvas.width * 0.94,
  x_text = canvas.width / 4;;
  ctx.font = "18px Courier New";
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

function arrayMin(arr)
{
  let min_val = 1e20;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < min_val){
      min_val = arr[i];
    }
  }
  return min_val
}

function arrayMax(arr)
{
  let max_val = -1e20;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > max_val){
      max_val = arr[i];
    }
  }
  return max_val
}

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
