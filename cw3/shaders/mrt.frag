#version 450
precision highp float;

layout(location = 0) in vec2 texCoords;
layout(location = 1) in vec3 worldPos;
layout(location = 2) in vec3 normal;


layout(push_constant) uniform PushConstantData {
    vec4 cameraPos;
} pushConstant;

layout(set = 1, binding = 0) uniform Colors {
    vec4 basecolor;
    vec4 emissive;
    float roughness;
    float metalness;
} colors;

layout(set = 2,binding = 0) uniform ULight{
	vec4 position;
	vec4 color;
}uLight;

layout(set = 3,binding = 0) uniform sampler2D albedoMap;
layout(set = 3,binding = 1) uniform sampler2D metallicMap;
layout(set = 3,binding = 2) uniform sampler2D roughnessMap;   

const float PI = 3.14159265359;

layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oNormal;
layout(location = 2) out vec4 oPosition;
layout(location = 3) out vec4 oBrightness;



float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness*roughness;
	float a2 = a*a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH*NdotH;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;
	return a2 / denom;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main()
{
    vec3 n = normalize(normal);
    // view direction, point to the camera
    vec3 v = normalize(vec3(pushConstant.cameraPos) - worldPos);
    vec3 l = normalize(uLight.position.xyz - worldPos);
    vec3 h = normalize(v + l);  

    float ndoth = clamp(dot(n, h), 0.0 ,1.0);
    float ndotv = clamp(dot(n, v), 0.0 ,1.0);
    float ndotl = clamp(dot(n, l), 0.0 ,1.0);
    float ldoth = clamp(dot(l, h), 0.0, 1.0);
    float vdoth = dot(v, h);
    
    vec3 basecolor = vec3(texture(albedoMap,texCoords)) * vec3(colors.basecolor);
    float metalliccolor = texture(metallicMap,texCoords).r * colors.metalness;
    float roughnesscolor = texture(roughnessMap,texCoords).r * colors.roughness;

    vec3 F0 = mix(vec3(0.04), vec3(basecolor),colors.metalness);

    vec3 color = vec3(0.0);

    if (ndotl > 0.0){
    vec3 F = fresnelSchlick(vdoth,F0);
    //Fresnel factor

    vec3 Ldiffuse = ( vec3(basecolor)/ PI ) * ( vec3(1) - F ) * ( 1 - metalliccolor);
   

	float Dh =  DistributionGGX(n,h,roughnesscolor);
   
    float Glv = min( 1 , min( 2 * ( ndoth * ndotv ) / vdoth , 2 * ( ndoth * ndotl)  / vdoth ) );
      
   
    vec3 specular = F * Dh * Glv / max(0.000001, 4.0 * ndotl * ndotv);
   
    color += (specular +  Ldiffuse) * ndotl + basecolor * 0.01 ;
    }

   	float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
	vec3 scaledColor = color / (luminance + 1.0);
    vec3 mappedColor = scaledColor / (scaledColor + 1.0);
    oColor = vec4(mappedColor, 1.0);
     
    float luminance2 = dot(colors.emissive.rgb, vec3(0.2126, 0.7152, 0.0722));
	vec3 scaledColor2 = colors.emissive.rgb / (luminance2 + 1.0);
    vec3 mappedColor2 = scaledColor2 / (scaledColor2 + 1.0);
    oBrightness = vec4(mappedColor2, 1.0);
    oNormal = vec4(normal,1.0);
    oPosition = vec4(worldPos,1.0);
}