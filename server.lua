-- load namespace
local socket = require("socket")
-- create a TCP socket and bind it to the local host, at any port
local server = assert(socket.bind("*", 51111))
-- find out which port the OS chose for us
local ip, port = server:getsockname()
-- print a message informing what's up
print("Please telnet to localhost on port " .. port)
print("After connecting, you have 10s to enter a line to be echoed")
-- loop forever waiting for clients

local client = server:accept()

while 1 do
  -- wait for a connection from any client
  -- make sure we don't block waiting for this client's line
  -- receive the line
  local line, err = client:receive()
  print(line)

  client:send("fish")

end