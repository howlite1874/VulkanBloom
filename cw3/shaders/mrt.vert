#version 450
layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexcoord;
layout(location = 2) in vec3 iNormals;

//std140
layout(set = 0,binding = 0) uniform UScene{
	mat4 camera;
	mat4 projection;
	mat4 projCamera;
}uScene;

layout(location = 0) out vec2 texCoords;
layout(location = 1) out vec3 worldPos;
layout(location = 2) out vec3 normal;

void main()
{
	texCoords = iTexcoord;
	worldPos = iPosition;
	normal = iNormals;

	gl_Position = uScene.projCamera * vec4(iPosition,1.f);

}

