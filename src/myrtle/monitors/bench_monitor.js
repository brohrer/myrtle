function GetURLParameter(sParam)
{
    var sPageURL = window.location.search.substring(1);
    var sURLVariables = sPageURL.split('&');
    for (var i = 0; i < sURLVariables.length; i++)
    {
        var sParameterName = sURLVariables[i].split('=');
        if (sParameterName[0] == sParam)
        {
            return sParameterName[1];
        }
    }
}

var host = GetURLParameter('host');
var port = GetURLParameter('port');

const addr = `ws://${host}:${port}`;
const socket = new WebSocket(addr);

// get canvas and context
var canvas = document.getElementById('the-canvas'),
ctx = canvas.getContext('2d');

// some kind of state for the animation
const x_center = canvas.width / 2,
y_center = canvas.width / 2,
dy_text = 0.07 * canvas.width,
radius = canvas.width / 4;

var msg = '',
reward = 0.0,
angle = 0.0,
velocity = 0.0,
step = 0,
episode = 0;

//socket.onopen = () => {
//  console.log('ws opened on browser');
//  socket.send('hello world');
//};

socket.onmessage = (event) => {
  console.log(event.data);
  x = 100;
  obj = JSON.parse(event.data);
  if (obj.message !== ""){
      msg = obj.message;
      values = JSON.parse(msg);
      sensor_array = values.sensors;
      step = values.loop_step;
      episode = values.episode;
      reward = values.rewards[0];
      //i_sensor = sensor_array.indexOf(1.0);
      //position = i_sensor * 10.0;
      angle = sensor_array[0];
      velocity = sensor_array[1];
  };
  event.data = angle;
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

    var y_text = canvas.width * 0.94,
    x_text = canvas.width / 4;;
    ctx.font = "18px Courier New";
    ctx.fillText(`reward     ${reward.toFixed(3)}`, x_text, y_text);
    ctx.fillText(`angle      ${angle.toFixed(2)}`, x_text, y_text + dy_text * 1);
    if (velocity < 0){
      ctx.fillText(`velocity  ${velocity.toFixed(2)}`, x_text, y_text + dy_text * 2);
    } else {
      ctx.fillText(`velocity   ${velocity.toFixed(2)}`, x_text, y_text + dy_text * 2);
    };
    ctx.fillText(`step       ${step}`, x_text, y_text + dy_text * 3);
    ctx.fillText(`episode    ${episode}`, x_text, y_text + dy_text * 4);

    //request next frame
    requestAnimationFrame(loop);
};

loop()
