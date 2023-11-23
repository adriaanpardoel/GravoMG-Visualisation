#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec4 aColor;

out vec4 ourColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec4 uColor;
uniform bool useUniformColor;

void main()
{
   gl_Position = projection * view * model * vec4(aPos, 1.0f);
   ourColor = useUniformColor ? uColor : aColor;
}
