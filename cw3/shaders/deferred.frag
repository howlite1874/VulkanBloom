#version 450
precision highp float;

layout (set = 0,binding = 0) uniform sampler2D samplerAlbedo;
layout (set = 0,binding = 1) uniform sampler2D samplerNormal;
layout (set = 0,binding = 2) uniform sampler2D samplerPosition;
layout (set = 0,binding = 3) uniform sampler2D samplerBrightness;


layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragcolor;

void main() 
{
	// Get G-Buffer values
	vec3 fragPos = texture(samplerPosition, inUV).rgb;
	vec3 normal = texture(samplerNormal, inUV).rgb;
	vec4 albedo = texture(samplerAlbedo, inUV);
	vec4 brightness = texture(samplerBrightness, inUV);
	outFragcolor = vec4(brightness.rgb + albedo.rgb, 1.0);
}