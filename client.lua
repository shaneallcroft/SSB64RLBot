local host, port = "127.0.0.1", 51111
local socket = require("socket")
local tcp = assert(socket.tcp())

tcp:connect(host, port);
--note the newline below
tcp:send("hello world\n");
local thing = "dog"
print(thing)



local thing2, err = tcp:receive("*l")

print(thing2)

tcp:close()