import * as config from './config.js';
import * as art from './drawingTools.js'
import * as num from './num.js';

const addr = `ws://${config.mq_host}:${config.mq_port}`;
const socket = new WebSocket(addr);

let canvas = document.getElementById('the-canvas'),
ctx = canvas.getContext('2d');

// Placement of the animation and text
const xCenterPixel = canvas.width / 2;
const yCenterPixel = canvas.width / 2;
const textYOffset = 0.07 * canvas.width;
const radius = canvas.width / 4;

// Determines whether an update to the display is needed.
// Refreshes to true every time a non-empty message is received.
let redraw = true;

let msg = '';
let reward = 0.0;
let angle = 0.0;
let velocity = 0.0;
let step = 0;
let episode = 0;

loop();

function render() {
  ctx.fillStyle = 'Black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'CadetBlue';
  ctx.beginPath();
  ctx.arc(
    xCenterPixel + radius * Math.sin(angle),
    yCenterPixel + radius * Math.cos(angle),
    10,
    0,
    Math.PI * 2
  );
  ctx.fill();

  let yText = canvas.width * 0.94;
  let xText = canvas.width / 4;
  ctx.font = "20px Courier New";
  ctx.fillText(`reward     ${reward.toFixed(3)}`, xText, yText);
  ctx.fillText(`angle      ${angle.toFixed(2)}`, xText, yText + textYOffset * 1);
  if (velocity < 0){
    ctx.fillText(
      `velocity  ${velocity.toFixed(2)}`,
      xText,
      yText + textYOffset * 2
    );
  } else {
    ctx.fillText(
      `velocity   ${velocity.toFixed(2)}`,
      xText,
      yText + textYOffset * 2
    );
  };
  ctx.fillText(`step       ${step}`, xText, yText + textYOffset * 3);
  ctx.fillText(`episode    ${episode}`, xText, yText + textYOffset * 4);

  redraw = false;
}

socket.onmessage = (event) => {
  let obj = JSON.parse(event.data);
  if (obj.message !== ""){
    redraw = true;
    msg = obj.message;
    let values = JSON.parse(msg);
    let sensor_array = values.sensors;
    step = values.loop_step;
    episode = values.episode;
    // TODO: sum all rewards
    reward = values.rewards[0];
    //i_sensor = sensor_array.indexOf(1.0);
    //position = i_sensor * 10.0;
    angle = sensor_array[0];
    velocity = sensor_array[1];
  };
};

// Main APP loop
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
