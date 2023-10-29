#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in int inSampling;

out vec3 ourColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 samplingColor;
uniform vec3 selectionColor;
uniform int selectedVertex;

void main()
{
   gl_Position = projection * view * model * vec4(aPos, 1.0f);
   ourColor = gl_VertexID == selectedVertex ? selectionColor : (inSampling == 1 ? samplingColor : vec3(0));
}
