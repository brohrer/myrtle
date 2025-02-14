//import { arrayMin, arrayMax } from './num.js';
import { mq_host, mq_port } from './config.js';

const addr = `ws://${mq_host}:${mq_port}`;
const socket = new WebSocket(addr);

// get canvas and context
let canvas = document.getElementById('the-canvas'),
ctx = canvas.getContext('2d');

// some kind of state for the animation
const x_center = canvas.width / 2,
y_center = canvas.width / 2,
dy_text = 0.07 * canvas.width,
radius = canvas.width / 4;

let msg = '';
let reward = 0.0;
let angle = 0.0;
let velocity = 0.0;
let step = 0;
let episode = 0;

socket.onmessage = (event) => {
  console.log(event.data);
  let obj = JSON.parse(event.data);
  if (obj.message !== ""){
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
var loop = function () {
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
    ctx.beginPath();
    ctx.arc(
      x_center + radius * Math.sin(angle),
      y_center + radius * Math.cos(angle),
      10,
      0,
      Math.PI * 2
    );
    ctx.fill();

    let y_text = canvas.width * 0.94;
    let x_text = canvas.width / 4;
    ctx.font = "24px Courier New";
    ctx.fillText(`reward     ${reward.toFixed(3)}`, x_text, y_text);
    ctx.fillText(`angle      ${angle.toFixed(2)}`, x_text, y_text + dy_text * 1);
    if (velocity < 0){
      ctx.fillText(
        `velocity  ${velocity.toFixed(2)}`,
        x_text,
        y_text + dy_text * 2
      );
    } else {
      ctx.fillText(
        `velocity   ${velocity.toFixed(2)}`,
        x_text,
        y_text + dy_text * 2
      );
    };
    ctx.fillText(`step       ${step}`, x_text, y_text + dy_text * 3);
    ctx.fillText(`episode    ${episode}`, x_text, y_text + dy_text * 4);

    //request next frame
    requestAnimationFrame(loop);
};

loop()
