#include <volk/volk.h>

#include <tuple>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>
#include <stb_image_write.h>
#include <glm/gtc/quaternion.hpp>
#include <array>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "../labutils/vertex_data.hpp"

#include "baked_model.hpp"


namespace
{
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
#		define SHADERDIR_ "assets/cw3/shaders/"

		constexpr char const* mrtVertPath = SHADERDIR_ "mrt.vert.spv";
		constexpr char const* mrtFragPath = SHADERDIR_ "mrt.frag.spv";

		constexpr char const* deferredVertPath = SHADERDIR_ "deferred.vert.spv";
		constexpr char const* deferredFragPath = SHADERDIR_ "deferred.frag.spv";

		constexpr char const* bloomVertPath = SHADERDIR_ "gaussblur.vert.spv";
		constexpr char const* bloomFragPath = SHADERDIR_ "gaussblur.frag.spv";

		//baked obj file
		constexpr char const* MODEL_PATH = "assets/cw3/ship.comp5822mesh";

#		undef SHADERDIR_

		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.1f;
		constexpr float kCameraFar   = 100.f;

		constexpr auto kCameraFov    = 60.0_degf;

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

		constexpr float kCameraBaseSpeed = 1.7f;
		constexpr float kCameraFastMult = 1.7f;
		constexpr float kCameraSlowMult = 1.7f;

		constexpr float kCameraMouseSensitivity = 0.1f;

		constexpr float kLightRotationSpeed = 0.1f;
	}

	using clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;



	// GLFW callbacks
	void glfw_callback_key_press( GLFWwindow*, int, int, int, int );

	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	// Uniform data
	namespace glsl
	{
		struct SceneUniform
		{
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCamera;
		};

		struct LightUniform
		{
			glm::vec4 position;
			glm::vec4 color;
		};

		struct ColorUniform
		{
			glm::vec4 basecolor;
			glm::vec4 emissive;
			float roughness;
			float metalness;
		};

		struct BloomUniform
		{
			float blurScale;
			float blurStrength;
		};

	}

	// Helpers:

	enum class EInputState
	{
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		rotateLight,
		max
	};

	struct  UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;

		bool wasMousing = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();

		glm::vec3 lightPosition = {0,10,0};
		glm::vec3 lightRotationCenter = {0,9.9999,0};
		float lightAngle;
	};

	// Framebuffer for offscreen rendering
	//----------------------------------------------------------------
	struct FrameBufferAttachment {
		lut::Image image;
		lut::ImageView view;
		VkFormat format;
	};
	//---------------------------------------------------------------
	struct offcreenFramebuffer
	{
		uint32_t width, height;
		lut::Framebuffer framebuffer;
		FrameBufferAttachment position, normal, albedo, brightness;
		FrameBufferAttachment depth;
		lut::RenderPass renderPass;
	}osFrameBuffer;

	//bloom pass
	struct FrameBuffer {
		lut::Framebuffer framebuffer;
		FrameBufferAttachment color, depth;
		VkDescriptorImageInfo descriptor;
	};
	struct OffscreenPass {
		int32_t width, height;
		lut::RenderPass renderPass;
		lut::Sampler sampler;
		std::array < FrameBuffer, 2 > framebuffers;
	} bloomPass;

	void update_user_state(UserState&, float aElapsedTime);

	lut::RenderPass create_render_pass( lut::VulkanWindow const& );

	lut::DescriptorSetLayout create_vert_descriptor_layout( lut::VulkanWindow const& );
	lut::DescriptorSetLayout create_frag_descriptor_layout( lut::VulkanWindow const& );
	lut::DescriptorSetLayout create_light_descriptor_layout( lut::VulkanWindow const& );
	lut::DescriptorSetLayout create_blur_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_texture_descriptor_layout(lut::VulkanWindow const& );
	lut::DescriptorSetLayout create_deferred_texture_descriptor_layout(lut::VulkanWindow const&);


	lut::PipelineLayout create_offscreen_pipeline_layout( lut::VulkanContext const& , VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout,
		VkDescriptorSetLayout aLightLayout, VkDescriptorSetLayout aTextureLayout);
	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& , VkDescriptorSetLayout aLayout);
	lut::PipelineLayout create_bloom_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout aLayout);



	lut::Pipeline create_offscreen_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_vertical_bloom_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_horizontal_bloom_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);



	//be used to create different loaded obj pipeline

	void create_swapchain_framebuffers( 
		lut::VulkanWindow const&, 
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState
	);





	void record_commands(
		VkCommandBuffer,
		offcreenFramebuffer &,
		VkExtent2D const&,
		VkBuffer aSceneUbo,
		VkBuffer aLightUbo,
		VkBuffer aBloomUbo,
		glsl::SceneUniform const&,
		glsl::LightUniform const&,
		glsl::BloomUniform const&,
		VkPipelineLayout,
		VkDescriptorSet aSceneDescriptors,
		VkDescriptorSet aLightDescriptors,
		std::vector<VkDescriptorSet> aTextureDescriptors,
		std::vector<objMesh>&&,
		std::vector<VkDescriptorSet> aObjDescriptors,
		VkPipeline,
		BakedModel const&,
		UserState,
		VkRenderPass,
		VkFramebuffer,
		VkDescriptorSet,
		VkPipelineLayout,
		VkPipeline,
		OffscreenPass&,
		VkPipelineLayout,
		VkPipeline,
		VkPipeline,
		VkDescriptorSet aBloomDescriptors,
		VkDescriptorSet aBloomDescriptors2
		//VkQueryPool aQueryPool
	);
	 
	void submit_commands(
		lut::VulkanWindow const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
	void present_results( 
		VkQueue, 
		VkSwapchainKHR, 
		std::uint32_t aImageIndex, 
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);


	void createColorUBOs(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator, std::vector<glsl::ColorUniform> colorUniforms, std::vector<lut::Buffer> &colorUBOs, BakedModel model);
	void createAttachment(VkFormat, VkImageUsageFlags,FrameBufferAttachment*,lut::VulkanWindow const&,labutils::Allocator const& );
	void createOffscreenFramebuffer(lut::VulkanWindow const&, labutils::Allocator const&);
	void createBloomFramebuffer(lut::VulkanWindow const&, labutils::Allocator const&, FrameBuffer*, VkFormat, VkFormat);
	void bloomPrepare(lut::VulkanWindow const&, labutils::Allocator const&);

}


int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();
	
	UserState state{};

	glfwSetWindowUserPointer(window.window, &state);
	glfwSetKeyCallback(window.window, &glfw_callback_key_press); 
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button); 
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);

	// Configure the GLFW window
	glfwSetKeyCallback( window.window, &glfw_callback_key_press );

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator( window );

	// final renderPass
	lut::RenderPass renderPass = create_render_pass( window );

	//create vert-stage descriptor set layout
	lut::DescriptorSetLayout sceneLayout = create_vert_descriptor_layout(window);

	//create frag-stage descriptor set layout
	lut::DescriptorSetLayout objectLayout = create_frag_descriptor_layout(window);

	lut::DescriptorSetLayout textureLayout = create_texture_descriptor_layout(window);

	lut::DescriptorSetLayout deferredTextureLayout = create_deferred_texture_descriptor_layout(window);

	lut::DescriptorSetLayout lightLayout = create_light_descriptor_layout(window);

	lut::DescriptorSetLayout blurLayout = create_blur_descriptor_layout(window);

	lut::PipelineLayout offscreenPipeLayout = create_offscreen_pipeline_layout( window , sceneLayout.handle,objectLayout.handle,lightLayout.handle, textureLayout.handle);
	lut::PipelineLayout pipeLayout = create_pipeline_layout(window,deferredTextureLayout.handle);
	lut::PipelineLayout bloomLayout = create_bloom_pipeline_layout(window, blurLayout.handle);

	//the process of creating depth buffer kind of like create an image,sowe also need an
	//image view(depth buffer view)
	auto[depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	std::vector<lut::Framebuffer> framebuffers;
	//similar to an image stored in the swap chain
	create_swapchain_framebuffers( window, renderPass.handle, framebuffers,depthBufferView.handle);

	// offscreen framebuffer and render pass
	createOffscreenFramebuffer(window, allocator);
	bloomPrepare(window, allocator);

	//pipeline with depth test
	//lut::Pipeline alphaPipe = create_alpha_pipeline(window, osFrameBuffer.renderPass.handle, pipeLayout.handle);
	lut::Pipeline offscreenPipe = create_offscreen_pipeline(window, osFrameBuffer.renderPass.handle, offscreenPipeLayout.handle);
	lut::Pipeline normalPipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);
	lut::Pipeline bloomVerticalPipe = create_vertical_bloom_pipeline(window, bloomPass.renderPass.handle, bloomLayout.handle);
	lut::Pipeline bloomHorizontalPipe = create_horizontal_bloom_pipeline(window, bloomPass.renderPass.handle, bloomLayout.handle);



	//create descriptor pool(all the descriptor set)
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	/*VkQueryPoolCreateInfo queryPoolCreateInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
	queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
	queryPoolCreateInfo.queryCount = 2;

	VkQueryPool measurePool;
	vkCreateQueryPool(window.device, &queryPoolCreateInfo, nullptr, &measurePool);*/

	//A VkCommandPool is required to be able to allocate a VkCommandBuffer.
	//for swapchain
	lut::CommandPool cpool = lut::create_command_pool( window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT );

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;

	//There are as many framebuffers as there are command buffer and as many fence
	for( std::size_t i = 0; i < framebuffers.size(); ++i )
	{
		cbuffers.emplace_back( lut::alloc_command_buffer( window, cpool.handle ) );
		cbfences.emplace_back( lut::create_fence( window, VK_FENCE_CREATE_SIGNALED_BIT ) );
	}

	lut::Semaphore imageAvailable = lut::create_semaphore( window );
	lut::Semaphore renderFinished = lut::create_semaphore( window );

	//------------------------------------------------------------------------------------------------------------------------------
	BakedModel model = load_baked_model(cfg::MODEL_PATH);

	//1.Create and load textures.This gives a list of Images(which includes a
	//VkImage + VmaAllocation) and VkImageViews.We only need to keep these
	//around -- place them in a vector.

	//load textures into image
	lut::CommandPool loadCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	std::vector<lut::Image> objTextures;
	for (const auto& t : model.textures)
	{
		lut::Image oneObjTex; 
		oneObjTex = lut::load_image_texture2d(
			t.path.c_str(), window,
			loadCmdPool.handle, allocator);
		objTextures.emplace_back(std::move(oneObjTex));
	}

	//create image view for texture image
	std::vector<lut::ImageView> objViews;
	for (size_t i = 0; i < model.textures.size(); i++)
	{
		lut::ImageView oneObjView;
		oneObjView = lut::create_image_view_texture2d(window, objTextures[i].image, VK_FORMAT_R8G8B8A8_SRGB);
		objViews.emplace_back(std::move(oneObjView));
	}

	//create texture sampler
	lut::Sampler defaultSampler = lut::create_default_sampler(window);
	lut::Sampler offscreenSampler = lut::create_deferred_sampler(window);
    //-----------------------------------------------------------------------------------------------------
	//scene uniform buffer in vertex shader
	lut::Buffer sceneUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer lightUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::LightUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer bloomUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::BloomUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	//-----------------------------------------------------------------------------------------------------------
	std::vector<glsl::ColorUniform> colorUniforms;
	colorUniforms.resize(model.meshes.size());
	for (uint32_t i = 0; i < model.meshes.size(); i++) {
		colorUniforms[i].basecolor = glm::vec4(model.materials[model.meshes[i].materialId].baseColor,1.0f);
		colorUniforms[i].emissive = glm::vec4(model.materials[model.meshes[i].materialId].emissiveColor, 1.0f);
		colorUniforms[i].metalness = model.materials[model.meshes[i].materialId].metalness;
		colorUniforms[i].roughness = model.materials[model.meshes[i].materialId].roughness;
	}

	std::vector<lut::Buffer> colorUBOs;
	createColorUBOs(window, allocator, colorUniforms, colorUBOs,model);

	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);

	//initialize descriptor set sceneUBO with vkUpdateDescriptorSets
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	VkDescriptorSet lightDescriptors = lut::alloc_desc_set(window, dpool.handle, lightLayout.handle);

	//initialize descriptor set sceneUBO with vkUpdateDescriptorSets
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo lightUboInfo{};
		lightUboInfo.buffer = lightUBO.buffer;
		lightUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = lightDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &lightUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	//--deferred texture descriptor--------------------------------------------------------------------------------
	VkDescriptorSet deferredDescriptors = lut::alloc_desc_set(window, dpool.handle, deferredTextureLayout.handle);
	
	VkWriteDescriptorSet deferreddesc[4]{};
	VkDescriptorImageInfo deferredtextureInfo[3]{};
	deferredtextureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	deferredtextureInfo[0].imageView = osFrameBuffer.albedo.view.handle;
	deferredtextureInfo[0].sampler = offscreenSampler.handle;

	deferredtextureInfo[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	deferredtextureInfo[1].imageView = osFrameBuffer.normal.view.handle;
	deferredtextureInfo[1].sampler = offscreenSampler.handle;


	deferredtextureInfo[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	deferredtextureInfo[2].imageView = osFrameBuffer.position.view.handle;
	deferredtextureInfo[2].sampler = offscreenSampler.handle;

	deferreddesc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	deferreddesc[0].dstSet = deferredDescriptors;
	deferreddesc[0].dstBinding = 0;
	deferreddesc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	deferreddesc[0].descriptorCount = 1;
	deferreddesc[0].pImageInfo = &deferredtextureInfo[0];

	deferreddesc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	deferreddesc[1].dstSet = deferredDescriptors;
	deferreddesc[1].dstBinding = 1;
	deferreddesc[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	deferreddesc[1].descriptorCount = 1;
	deferreddesc[1].pImageInfo = &deferredtextureInfo[1];

	deferreddesc[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	deferreddesc[2].dstSet = deferredDescriptors;
	deferreddesc[2].dstBinding = 2;
	deferreddesc[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	deferreddesc[2].descriptorCount = 1;
	deferreddesc[2].pImageInfo = &deferredtextureInfo[2];

	deferreddesc[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	deferreddesc[3].dstSet = deferredDescriptors;
	deferreddesc[3].dstBinding = 3;
	deferreddesc[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	deferreddesc[3].descriptorCount = 1;
	deferreddesc[3].pImageInfo = &bloomPass.framebuffers[1].descriptor;
	//--------------------------------------------------------------------------------
	// Full screen blur
	// Vertical-----------------------------------------------------------------------------------------------
	VkDescriptorImageInfo bloomTextureInfo{};
	bloomTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	bloomTextureInfo.imageView = osFrameBuffer.brightness.view.handle;
	bloomTextureInfo.sampler = offscreenSampler.handle;

	VkDescriptorBufferInfo bloomUboInfo{};
	bloomUboInfo.buffer = bloomUBO.buffer;
	bloomUboInfo.range = VK_WHOLE_SIZE;

	VkDescriptorSet bloomDescriptors = lut::alloc_desc_set(window, dpool.handle, blurLayout.handle);
	VkWriteDescriptorSet bloomdesc[2]{};

	bloomdesc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	bloomdesc[0].dstSet = bloomDescriptors;
	bloomdesc[0].dstBinding = 0;
	bloomdesc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	bloomdesc[0].descriptorCount = 1;
	bloomdesc[0].pBufferInfo = &bloomUboInfo;

	bloomdesc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	bloomdesc[1].dstSet = bloomDescriptors;
	bloomdesc[1].dstBinding = 1;
	bloomdesc[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	bloomdesc[1].descriptorCount = 1;
	bloomdesc[1].pImageInfo = &bloomTextureInfo;

	vkUpdateDescriptorSets(window.device, 2, bloomdesc, 0, nullptr);

	// Horizontal-------------------------------------------------------------------------------------------------
	VkDescriptorSet bloomDescriptors2 = lut::alloc_desc_set(window, dpool.handle, blurLayout.handle);
	VkWriteDescriptorSet bloomdesc2[2]{};

	bloomdesc2[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	bloomdesc2[0].dstSet = bloomDescriptors2;
	bloomdesc2[0].dstBinding = 0;
	bloomdesc2[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	bloomdesc2[0].descriptorCount = 1;
	bloomdesc2[0].pBufferInfo = &bloomUboInfo;

	bloomdesc2[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	bloomdesc2[1].dstSet = bloomDescriptors2;
	bloomdesc2[1].dstBinding = 1;
	bloomdesc2[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	bloomdesc2[1].descriptorCount = 1;
	bloomdesc2[1].pImageInfo = &bloomPass.framebuffers[0].descriptor;

	vkUpdateDescriptorSets(window.device, 2, bloomdesc2, 0, nullptr);

	//--------------------------------------------------------------------------------
	std::vector<VkDescriptorSet> objDescriptors;
	std::vector<VkDescriptorBufferInfo> colorUboInfos{};
	colorUboInfos.resize(model.meshes.size());
	for (uint32_t i = 0; i < model.meshes.size(); i++) {
		colorUboInfos[i].buffer = colorUBOs[i].buffer;
		colorUboInfos[i].range = VK_WHOLE_SIZE;
	}

	int count = 0;
	for (uint32_t i = 0; i < model.meshes.size(); i++)
	{
		VkDescriptorSet oneObjDescriptors = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);

		VkWriteDescriptorSet desc[1]{};

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = oneObjDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &colorUboInfos[count];


		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
		objDescriptors.emplace_back(oneObjDescriptors);
		count++;
	}

	std::vector<VkDescriptorSet> textureDescriptors;
	for (const auto& m : model.meshes)
	{
		VkDescriptorSet oneObjDescriptors = lut::alloc_desc_set(window, dpool.handle, textureLayout.handle);

		VkWriteDescriptorSet desc[3]{};
		std::uint32_t mid = m.materialId;
		VkDescriptorImageInfo textureInfo[3]{};
		textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo[0].imageView = objViews[model.materials[mid].baseColorTextureId].handle;
		textureInfo[0].sampler = defaultSampler.handle;

		textureInfo[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo[1].imageView = objViews[model.materials[mid].metalnessTextureId].handle;
		textureInfo[1].sampler = defaultSampler.handle;

		textureInfo[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo[2].imageView = objViews[model.materials[mid].roughnessTextureId].handle;
		textureInfo[2].sampler = defaultSampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = oneObjDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &textureInfo[0];

		desc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[1].dstSet = oneObjDescriptors;
		desc[1].dstBinding = 1;
		desc[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[1].descriptorCount = 1;
		desc[1].pImageInfo = &textureInfo[1];

		desc[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[2].dstSet = oneObjDescriptors;
		desc[2].dstBinding = 2;
		desc[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[2].descriptorCount = 1;
		desc[2].pImageInfo = &textureInfo[2];

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
		textureDescriptors.emplace_back(oneObjDescriptors);
	}
	

	//4.Upload mesh data.In my reference solution, I created separate VkBuffers
	//for each mesh(one for each attribute and one for the indices).
	//loda baked model and create buffer per mesh
	std::vector<objMesh> oMesh;
	for(auto const &m: model.meshes)
	{
		objMesh meshBuffer = create_mesh(window,allocator, m);
		oMesh.emplace_back(std::move(meshBuffer));
	}

	// Application main loop
	bool recreateSwapchain = false;
	auto previousClock = clock_::now();
	while( !glfwWindowShouldClose( window.window ) )
	{
		// Let GLFW process events.
		// glfwPollEvents() checks for events, processes them. If there are no
		// events, it will return immediately. Alternatively, glfwWaitEvents()
		// will wait for any event to occur, process it, and only return at
		// that point. The former is useful for applications where you want to
		// render as fast as possible, whereas the latter is useful for
		// input-driven applications, where redrawing is only needed in
		// reaction to user input (or similar).
		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if( recreateSwapchain )
		{
			//re-create swapchain and associated resources
			vkDeviceWaitIdle(window.device);

			//recreate them
			const auto changes = recreate_swapchain(window);

			if (changes.changedFormat) {
				renderPass = create_render_pass(window);
				createOffscreenFramebuffer(window, allocator);
				bloomPrepare(window, allocator);
			}

			if (changes.changedSize) {
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
				offscreenPipe = create_offscreen_pipeline(window, osFrameBuffer.renderPass.handle, offscreenPipeLayout.handle);
				normalPipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);
				bloomVerticalPipe = create_vertical_bloom_pipeline(window, bloomPass.renderPass.handle, bloomLayout.handle);
				bloomHorizontalPipe = create_horizontal_bloom_pipeline(window, bloomPass.renderPass.handle, bloomLayout.handle);

				createOffscreenFramebuffer(window, allocator);
				bloomPrepare(window, allocator);

			}

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

			recreateSwapchain = false;
			continue;
		}

		//acquire swapchain image
		std::uint32_t imageIndex = 0;
		const auto acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire enxt swapchain image\n"
				"vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str()
			);
		}

		//for deferred render
		// wait for command buffer to be available
		assert(std::size_t(imageIndex) < cbfences.size());
		if (const auto res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n"
				"vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str()
			);
		}

		if (const auto res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n"
				"vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str()
			);
		}


		auto const now = clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		update_user_state(state, dt);

		//record and submit commands(have done)
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);

		glsl::LightUniform lightUniforms{};
		lightUniforms.position = glm::vec4(state.lightPosition,1.0f);
		lightUniforms.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

		glsl::BloomUniform bloomUniforms{};
		bloomUniforms.blurScale = 1.1f;
		bloomUniforms.blurStrength = 1.5f;

		static_assert(sizeof(sceneUniforms) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(sceneUniforms) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");

		vkUpdateDescriptorSets(window.device, 4, deferreddesc, 0, nullptr);
		vkUpdateDescriptorSets(window.device, 2, bloomdesc, 0, nullptr);
		vkUpdateDescriptorSets(window.device, 2, bloomdesc2, 0, nullptr);

		record_commands(
		//for offscreen render
			cbuffers[imageIndex],
			osFrameBuffer,
			window.swapchainExtent,
			sceneUBO.buffer,
			lightUBO.buffer,
			bloomUBO.buffer,
			sceneUniforms,
			lightUniforms,
			bloomUniforms,
			offscreenPipeLayout.handle,
			sceneDescriptors,
			lightDescriptors,
			textureDescriptors,
			std::move(oMesh),
			objDescriptors,
			offscreenPipe.handle,
			model,
			state,
			renderPass.handle,
			framebuffers[imageIndex].handle,
			deferredDescriptors,
			pipeLayout.handle,
			normalPipe.handle,
			bloomPass,
			bloomLayout.handle,
			bloomVerticalPipe.handle,
			bloomHorizontalPipe.handle,
			bloomDescriptors,
			bloomDescriptors2
			//measurePool
		);

		submit_commands(
			window,
			cbuffers[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		// Retrieve the recorded timestamps from the query pool
		//uint64_t timestamps[2];
		//VkResult result = vkGetQueryPoolResults(window.device, measurePool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		//if (result != VK_SUCCESS) {
		//	// Handle error
		//}

		// Calculate the duration of the operation
		//uint64_t duration = timestamps[1] - timestamps[0];
		//std::cout << "vkCmdDrawIndexed took " << duration << " clock cycles\n";

		present_results(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);

	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.

	vkDeviceWaitIdle( window.device );

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

namespace
{
	void glfw_callback_key_press( GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/ )
	{
		if( GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction )
		{
			glfwSetWindowShouldClose( aWindow, GLFW_TRUE );
		}

		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		const bool isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;
		case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;
		case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
		case GLFW_KEY_SPACE:
			if (GLFW_PRESS == aAction)
			{
				state->inputMap[std::size_t(EInputState::rotateLight)] = !state->inputMap[std::size_t(EInputState::rotateLight)];
			}
			break;
		default:
			;
		}
	}

	void glfw_callback_button(GLFWwindow* aWin,int aBut,int aAct,int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if(GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	void glfw_callback_motion(GLFWwindow* aWin,double aX,double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}

	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;

		if(aState.inputMap[std::size_t(EInputState::mousing)])
		{
			if(aState.wasMousing)
			{
				const auto sens = cfg::kCameraMouseSensitivity;
				const auto dx = sens * (aState.mouseX - aState.previousX);
				const auto dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));
			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		}
		else
		{
			aState.wasMousing = false;
		}

		const auto move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f);

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, move));
		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move,0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move,0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));
		if (aState.inputMap[std::size_t(EInputState::rotateLight)])
		{
			float degreesPerSecond = 100.0f; 
			float rotationSpeed = glm::radians(degreesPerSecond);
			aState.lightAngle += rotationSpeed * aElapsedTime;

			float r = (aState.lightPosition-aState.lightRotationCenter).length();
			float theta = aState.lightAngle;
			float x = r * cos(theta);
			float z = r * sin(theta);
			aState.lightPosition = glm::vec3(x, aState.lightPosition.y, z);
		}
		
	}

}

namespace
{
	void update_scene_uniforms( glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight,UserState aState )
	{
		float const aspect = static_cast<float>(aFramebufferWidth) / aFramebufferHeight;
		//The RH indicates a right handed clip space, and the ZO indicates
		//that the clip space extends from zero to one along the Z - axis.
		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
		aSceneUniforms.projection[1][1] *= -1.f;
		aSceneUniforms.camera	  = glm::inverse(aState.camera2world);
		aSceneUniforms.projCamera = aSceneUniforms.projection * aSceneUniforms.camera;
	}
}

namespace
{
	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{
		//Create Render Pass Attachments
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		//Create Subpass Definition
		//The reference means it uses the attachment above as a color attachment
		VkAttachmentReference subpassAttachments[1]{};
		//the zero refers to the 0th render pass attachment declared earlier
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		//create render pass information
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0;
		passInfo.pDependencies = nullptr;

		//create render pass and see if the render pass is created successfully
		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str()
			);

		}

		//return the wrapped value
		return lut::RenderPass(aWindow.device, rpass);
	}


	void createAttachment(
		VkFormat format,
		VkImageUsageFlags usage,
		FrameBufferAttachment* attachment,
		lut::VulkanWindow const& aWindow,
		labutils::Allocator const& aAllocator
		)
	{
		attachment->format = format;

		if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
		{
			attachment->image = lut::create_image_texture2d(aAllocator, osFrameBuffer.width, osFrameBuffer.height, format, usage | VK_IMAGE_USAGE_SAMPLED_BIT);
			attachment->view = lut::create_image_view_texture2d(aWindow, attachment->image.image, format);
		}
		if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			auto [depthBuffer, depthBufferView] = create_depth_buffer(aWindow, aAllocator);
			attachment->image = std::move(depthBuffer);
			attachment->view = std::move(depthBufferView);
		}
	}


	void createOffscreenFramebuffer(lut::VulkanWindow const& aWindow, labutils::Allocator const& aAllocator)
	{
		osFrameBuffer.width = aWindow.swapchainExtent.width;
		osFrameBuffer.height = aWindow.swapchainExtent.height;

		// Create Attachments
		// Color attachment for albedo and emissive
		createAttachment(
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT ,
			&osFrameBuffer.albedo,
			aWindow,
			aAllocator
			);
		// Color attachment for normal
		createAttachment(
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
			&osFrameBuffer.normal,
			aWindow,
			aAllocator
		);
		// Color attachment for position
		createAttachment(
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
			&osFrameBuffer.position,
			aWindow,
			aAllocator
		);
		// Color attachment for brightness
		createAttachment(
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
			&osFrameBuffer.brightness,
			aWindow,
			aAllocator
		);

		// Depth-stencil attachment
		createAttachment(
			cfg::kDepthFormat,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			&osFrameBuffer.depth,
			aWindow,
			aAllocator
		);


		// Set up separate renderpass with references to the color and depth attachments
		std::array<VkAttachmentDescription, 5> attachmentDescs = {};

		// Init attachment properties
		for (uint32_t i = 0; i < 5; ++i)
		{
			attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
			attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			if (i == 4)
			{
				attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			}
			else
			{
				attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}
		}

		// Formats
		attachmentDescs[0].format = osFrameBuffer.albedo.format;
		attachmentDescs[1].format = osFrameBuffer.normal.format;
		attachmentDescs[2].format = osFrameBuffer.position.format;
		attachmentDescs[3].format = osFrameBuffer.brightness.format;
		attachmentDescs[4].format = osFrameBuffer.depth.format;

		// Create Subpass Definition
		VkAttachmentReference colorAttachments[4]{};
		// Color attachment for albedo and emissive
		colorAttachments[0].attachment = 0;
		colorAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		// Color attachment for normals
		colorAttachments[1].attachment = 1;
		colorAttachments[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		// Color attachment for position
		colorAttachments[2].attachment = 2;
		colorAttachments[2].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		// Color attachment for brightness
		colorAttachments[3].attachment = 3;
		colorAttachments[3].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 4;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses{};
		subpasses.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses.colorAttachmentCount = 4;
		subpasses.pColorAttachments = colorAttachments;
		subpasses.pDepthStencilAttachment = &depthAttachment;

		// Use subpass dependencies for attachment layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Create Render Pass Information
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
		passInfo.pAttachments = attachmentDescs.data();
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = &subpasses;
		passInfo.dependencyCount = 2;
		passInfo.pDependencies = dependencies.data();

		// Create Render Pass
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &osFrameBuffer.renderPass.handle); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to create render pass for deffered shading\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str()
			);
		}
		
		std::array<VkImageView, 5> attachments;
		attachments[0] = osFrameBuffer.albedo.view.handle;
		attachments[1] = osFrameBuffer.normal.view.handle;
		attachments[2] = osFrameBuffer.position.view.handle;
		attachments[3] = osFrameBuffer.brightness.view.handle;
		attachments[4] = osFrameBuffer.depth.view.handle;

		VkFramebufferCreateInfo fbufCreateInfo = {};
		fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbufCreateInfo.pNext = NULL;
		fbufCreateInfo.renderPass = osFrameBuffer.renderPass.handle;
		fbufCreateInfo.pAttachments = attachments.data();
		fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		fbufCreateInfo.width = osFrameBuffer.width;
		fbufCreateInfo.height = osFrameBuffer.height;
		fbufCreateInfo.layers = 1;

		if (auto const res = vkCreateFramebuffer(aWindow.device, &fbufCreateInfo, nullptr,&osFrameBuffer.framebuffer.handle); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to create framebuffer for deffered shading\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str()
			);
		}
	}

	// Setup the offscreen framebuffer for rendering the mirrored scene
	// The color attachment of this framebuffer will then be sampled from
	void createBloomFramebuffer(lut::VulkanWindow const& aWindow, labutils::Allocator const& aAllocator, FrameBuffer* frameBuf, VkFormat colorFormat, VkFormat depthFormat)
	{
		// Color attachment
		auto image = lut::create_image_texture2d(aAllocator, aWindow.swapchainExtent.width, aWindow.swapchainExtent.height, colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

		auto colorImageView = lut::create_image_view_texture2d(aWindow, image.image, colorFormat);

		auto [depthBuffer, depthBufferView] = create_depth_buffer(aWindow, aAllocator);

		frameBuf->color.view = std::move(colorImageView);
		frameBuf->color.image = std::move(image);
		frameBuf->depth.view = std::move(depthBufferView);
		frameBuf->depth.image = std::move(depthBuffer);
		frameBuf->color.format = colorFormat;
		frameBuf->depth.format = depthFormat;

		VkImageView attachments[2];
		attachments[0] = frameBuf->color.view.handle;
		attachments[1] = frameBuf->depth.view.handle;

		VkFramebufferCreateInfo fbufCreateInfo{};
		fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbufCreateInfo.renderPass = bloomPass.renderPass.handle;
		fbufCreateInfo.attachmentCount = 2;
		fbufCreateInfo.pAttachments = attachments;
		fbufCreateInfo.width = aWindow.swapchainExtent.width;
		fbufCreateInfo.height = aWindow.swapchainExtent.height;
		fbufCreateInfo.layers = 1;

		if (auto const res = vkCreateFramebuffer(aWindow.device, &fbufCreateInfo, nullptr,&frameBuf->framebuffer.handle); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to create framebuffer for deffered shading\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str()
			);
		}

		// Fill a descriptor for later use in a descriptor set
		frameBuf->descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		frameBuf->descriptor.imageView = frameBuf->color.view.handle;
		frameBuf->descriptor.sampler = bloomPass.sampler.handle;
	}

	// Prepare the offscreen framebuffers for the vertical and horizontal blur
	void bloomPrepare(lut::VulkanWindow const& aWindow, labutils::Allocator const& aAllocator)
	{
		bloomPass.width = aWindow.swapchainExtent.width;
		bloomPass.height = aWindow.swapchainExtent.height;

		// Create a separate render pass for the offscreen rendering as it may differ from the one used for scene rendering

		std::array<VkAttachmentDescription, 2> attchmentDescriptions = {};
		// Color attachment
		attchmentDescriptions[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
		attchmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attchmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attchmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attchmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attchmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attchmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		// Depth attachment
		attchmentDescriptions[1].format = cfg::kDepthFormat;
		attchmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attchmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attchmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attchmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attchmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
		VkAttachmentReference depthReference = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;
		subpassDescription.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Create the actual renderpass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
		renderPassInfo.pAttachments = attchmentDescriptions.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		// Create Render Pass
		if (auto const res = vkCreateRenderPass(aWindow.device, &renderPassInfo, nullptr, &bloomPass.renderPass.handle); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to create render pass for deffered shading\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str()
			);
		}

		// Create sampler to sample from the color attachments
		auto sampler = lut::create_default_sampler(aWindow);
		bloomPass.sampler = std::move(sampler);

		// Create two frame buffers
		createBloomFramebuffer(aWindow, aAllocator, &bloomPass.framebuffers[0], VK_FORMAT_R16G16B16A16_SFLOAT, cfg::kDepthFormat);
		createBloomFramebuffer(aWindow, aAllocator, &bloomPass.framebuffers[1], VK_FORMAT_R16G16B16A16_SFLOAT, cfg::kDepthFormat);
	}



	lut::PipelineLayout create_offscreen_pipeline_layout( lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout,
		VkDescriptorSetLayout aLightLayout, VkDescriptorSetLayout aTextureLayout)
	{
		//in two shader state,there are two layouts about what uniforms it has
		VkDescriptorSetLayout layouts[] = { 
			// Order must match the set = N in the shaders 
			aSceneLayout,   //set 0
			aObjectLayout,  //set 1
			aLightLayout,    //set 2
			aTextureLayout   //set 3
		};

		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(glm::vec4);

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		//return wrapped info
		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aLayout)
	{
		//in two shader state,there are two layouts about what uniforms it has
		VkDescriptorSetLayout layouts[] = {
			// Order must match the set = N in the shaders 
			aLayout   //set 0
		};

		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(glm::vec4);

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		//return wrapped info
		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_bloom_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aLayout)
	{
		//in two shader state,there are two layouts about what uniforms it has
		VkDescriptorSetLayout layouts[] = {
			// Order must match the set = N in the shaders 
			aLayout   //set 0
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = 1;
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		//return wrapped info
		return lut::PipelineLayout(aContext.device, layout);
	}


	lut::Pipeline create_offscreen_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		//load shader
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::mrtVertPath);

		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::mrtFragPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.back.compareOp = VK_COMPARE_OP_ALWAYS;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		VkVertexInputBindingDescription vertexInputs[3]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0;		//must match binding above
		vertexAttributes[0].location = 0;		//must match shader;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1;		//must match binding above
		vertexAttributes[1].location = 1;		//must match shader;
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2;		//must match binding above
		vertexAttributes[2].location = 2;		//must match shader;
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;
		                                                                                                                                
		inputInfo.vertexBindingDescriptionCount = 3;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		//input assembly state:define which primitive(point,line,triangle) the input is
		//assembled for rasterization
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//tessellation state:
		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		//scissor can be used to restrict drawing to a part of the frame buffer without changing the coordinate system
		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//rasterization state:
		//define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasClamp = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Multisample State£º
		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[4]{};
		for (int i = 0; i < 4; i++) {
			blendStates[i].colorWriteMask = 0xf;
			blendStates[i].blendEnable = VK_FALSE;
		}

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.attachmentCount = 4;
		blendInfo.pAttachments = blendStates;

		//dynamic state:none

		//Assembling the VkGraphicsPipelineCreateInfo structure
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;
		
		VkPipeline pipe = VK_NULL_HANDLE;
		//the second arguement means whether to use VkPipelineCache which can keep the cost down
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str()
			);
		}
		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		//load shader
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::deferredVertPath);

		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::deferredFragPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		//input assembly state:define which primitive(point,line,triangle) the input is
		//assembled for rasterization
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//tessellation state:
		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		//scissor can be used to restrict drawing to a part of the frame buffer without changing the coordinate system
		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//rasterization state:
		//define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_NONE;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasClamp = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Multisample State£º
		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_TRUE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		//In Vulkan, logical operations are used to perform bitwise operations on color data in a framebuffer
		//attachment during blending.(e.g. AND, OR, XOR)
		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;
		
		//dynamic state:none

		//Assembling the VkGraphicsPipelineCreateInfo structure
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		pipeInfo.pVertexInputState = &inputInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		//the second arguement means whether to use VkPipelineCache which can keep the cost down
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str()
			);
		}
		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_vertical_bloom_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		//load shader
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::bloomVertPath);

		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::bloomFragPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		// Use specialization constants to select between horizontal and vertical blur
		uint32_t blurdirection = 0;
		VkSpecializationMapEntry specializationMapEntry{};
		specializationMapEntry.constantID = 0;
		specializationMapEntry.offset = 0;
		specializationMapEntry.size = sizeof(uint32_t);

		VkSpecializationInfo specializationInfo{};
		specializationInfo.mapEntryCount = 1;
		specializationInfo.pMapEntries = &specializationMapEntry;
		specializationInfo.dataSize = sizeof(uint32_t);
		specializationInfo.pData = &blurdirection;

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";
		stages[1].pSpecializationInfo= &specializationInfo;




		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		//input assembly state:define which primitive(point,line,triangle) the input is
		//assembled for rasterization
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//tessellation state:
		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		//scissor can be used to restrict drawing to a part of the frame buffer without changing the coordinate system
		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//rasterization state:
		//define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_NONE;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasClamp = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Multisample State£º
		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendAttachmentState{};
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

		//In Vulkan, logical operations are used to perform bitwise operations on color data in a framebuffer
		//attachment during blending.(e.g. AND, OR, XOR)
		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = &blendAttachmentState;

		//dynamic state:none

		//Assembling the VkGraphicsPipelineCreateInfo structure
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		pipeInfo.pVertexInputState = &inputInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		//the second arguement means whether to use VkPipelineCache which can keep the cost down
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str()
			);
		}
		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_horizontal_bloom_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		//load shader
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::bloomVertPath);

		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::bloomFragPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		// Use specialization constants to select between horizontal and vertical blur
		uint32_t blurdirection = 1;
		VkSpecializationMapEntry specializationMapEntry{};
		specializationMapEntry.constantID = 0;
		specializationMapEntry.offset = 0;
		specializationMapEntry.size = sizeof(uint32_t);

		VkSpecializationInfo specializationInfo{};
		specializationInfo.mapEntryCount = 1;
		specializationInfo.pMapEntries = &specializationMapEntry;
		specializationInfo.dataSize = sizeof(uint32_t);
		specializationInfo.pData = &blurdirection;

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";
		stages[1].pSpecializationInfo = &specializationInfo;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		//input assembly state:define which primitive(point,line,triangle) the input is
		//assembled for rasterization
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//tessellation state:
		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		//scissor can be used to restrict drawing to a part of the frame buffer without changing the coordinate system
		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//rasterization state:
		//define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_NONE;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasClamp = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Multisample State£º
		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendAttachmentState{};
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

		//In Vulkan, logical operations are used to perform bitwise operations on color data in a framebuffer
		//attachment during blending.(e.g. AND, OR, XOR)
		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = &blendAttachmentState;

		//dynamic state:none

		//Assembling the VkGraphicsPipelineCreateInfo structure
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		pipeInfo.pVertexInputState = &inputInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		//the second arguement means whether to use VkPipelineCache which can keep the cost down
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str()
			);
		}
		return lut::Pipeline(aWindow.device, pipe);
	}


	void create_swapchain_framebuffers( lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers,VkImageView aDepthView )
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); i++) {
			VkImageView attachments[2] = {
				aWindow.swapViews[i],

				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; //normal frame buffer
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer\n"
					"vkCreateFramebuffer() returned %s", lut::to_string(res).c_str()
				);
			}
			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	lut::DescriptorSetLayout create_vert_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout{};
		if (const auto& res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_frag_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		//uniform sampler2D
		//color uniform buffer
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; /* image/texture2D sampler   */
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;     /* define as fragment shader */


		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_texture_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[3]{};
		//uniform sampler2D
		//1.base color
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; /* image/texture2D sampler   */
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;     /* define as fragment shader */

		//2.metalness color
		bindings[1].binding = 1;
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; /* image/texture2D sampler   */
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;     /* define as fragment shader */

		//3.roughness 
		bindings[2].binding = 2;
		bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; /* image/texture2D sampler   */
		bindings[2].descriptorCount = 1;
		bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;     /* define as fragment shader */

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_deferred_texture_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorBindingFlags binding_flags[4] = {
			VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
			VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
			VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
			VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		};

		VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
		binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
		binding_flags_info.bindingCount = sizeof(binding_flags) / sizeof(binding_flags[0]);
		binding_flags_info.pBindingFlags = binding_flags;

		VkDescriptorSetLayoutBinding bindings[4]{};
		//uniform sampler2D
		//1.base color
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; /* image/texture2D sampler   */
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;     /* define as fragment shader */

		//2.normal color
		bindings[1].binding = 1;
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; /* image/texture2D sampler   */
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;     /* define as fragment shader */
		 
		//3.position 
		bindings[2].binding = 2;
		bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; /* image/texture2D sampler   */
		bindings[2].descriptorCount = 1;
		bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;     /* define as fragment shader */

		//4.brightness
		bindings[3].binding = 3;
		bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; /* image/texture2D sampler   */
		bindings[3].descriptorCount = 1;
		bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;     /* define as fragment shader */

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT; // Add this line
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pNext = &binding_flags_info; 
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_light_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		//uniform 
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; 
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; 

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_blur_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorBindingFlags binding_flags[2] = {
	VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
	VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		};

		VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
		binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
		binding_flags_info.bindingCount = sizeof(binding_flags) / sizeof(binding_flags[0]);
		binding_flags_info.pBindingFlags = binding_flags;

		VkDescriptorSetLayoutBinding bindings[2]{};
		//uniform 
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		//image sampler
				//uniform 
		bindings[1].binding = 1;
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT; 
		layoutInfo.pNext = &binding_flags_info;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (const auto res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	void record_commands(VkCommandBuffer aCmdBuff, offcreenFramebuffer& aOSframebuffer, VkExtent2D const& aImageExtent, VkBuffer aSceneUbo, VkBuffer aLightUbo, VkBuffer aBloomUbo,
		glsl::SceneUniform const& aSceneUniform, glsl::LightUniform const& aLightUniform, glsl::BloomUniform const& aBloomUniform, VkPipelineLayout aGraphicsLayout, VkDescriptorSet aSceneDescriptors, VkDescriptorSet aLightDescriptors,
		std::vector<VkDescriptorSet> aTextureDescriptors, std::vector<objMesh>&& aObjMesh, std::vector<VkDescriptorSet> aObjDescriptors, VkPipeline aAlphaPipe, BakedModel const& aModel, UserState aState,
		VkRenderPass aRenderPass, VkFramebuffer aFramebuffer, VkDescriptorSet aOffscreenDescriptors, VkPipelineLayout aNormalLayout, VkPipeline aPipeline, OffscreenPass& aBloomPass,VkPipelineLayout aBloomLayout,
		VkPipeline aVerticalBloomPipe, VkPipeline aHorizontalBloomPipe, VkDescriptorSet aBloomDescriptors,VkDescriptorSet aBloomDescriptors2//,VkQueryPool aMeasurePool
	)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		//begin recording commands
		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &beginInfo);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}

		lut::buffer_barrier(aCmdBuff,
			aSceneUbo,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aSceneUbo, 0, sizeof(glsl::SceneUniform), &aSceneUniform);

		lut::buffer_barrier(aCmdBuff,
			aSceneUbo,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		lut::buffer_barrier(aCmdBuff,
			aLightUbo,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aLightUbo, 0, sizeof(glsl::LightUniform), &aLightUniform);

		lut::buffer_barrier(aCmdBuff,
			aLightUbo,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		lut::buffer_barrier(aCmdBuff,
			aBloomUbo,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aBloomUbo, 0, sizeof(glsl::BloomUniform), &aBloomUniform);

		lut::buffer_barrier(aCmdBuff,
			aBloomUbo,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		);

		//begin render pass
		VkClearValue clearValues[5]{};
		// Clear to a dark gray background. If we were debugging, this would potentially 
		// help us see whether the render pass took place, even if nothing else was drawn
		for (int i = 0; i < 4; i++) {
			clearValues[i].color.float32[0] = 0.0f;
			clearValues[i].color.float32[1] = 0.0f;
			clearValues[i].color.float32[2] = 0.0f;
			clearValues[i].color.float32[3] = 1.0f;
		}

		clearValues[4].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aOSframebuffer.renderPass.handle;
		passInfo.framebuffer = aOSframebuffer.framebuffer.handle;
		passInfo.renderArea.offset = VkOffset2D{ 0,0 };
		passInfo.renderArea.extent = VkExtent2D{ aImageExtent.width,aImageExtent.height };
		passInfo.clearValueCount = 5;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
		glm::vec4 cameraPos = aState.camera2world[3];
		vkCmdPushConstants(aCmdBuff, aGraphicsLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec4), &cameraPos);
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aAlphaPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 2, 1, &aLightDescriptors, 0, nullptr);
		for (uint32_t i = 0; i < aObjMesh.size(); i++) {
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, &aObjDescriptors[i], 0, nullptr);
			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 3, 1, &aTextureDescriptors[i], 0, nullptr);

			//Bind vertex input
			VkBuffer objBuffers[3] = { aObjMesh[i].positions.buffer, aObjMesh[i].texcoords.buffer, aObjMesh[i].normals.buffer
			};
			VkDeviceSize objOffsets[3]{};

			vkCmdBindVertexBuffers(aCmdBuff, 0, 3, objBuffers, objOffsets);
			vkCmdBindIndexBuffer(aCmdBuff, aObjMesh[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);

			vkCmdDrawIndexed(aCmdBuff, static_cast<uint32_t>(aModel.meshes[i].indices.size()), 1, 0, 0, 0);
		}
		//end the render pass
		 vkCmdEndRenderPass(aCmdBuff);
		 //vkCmdWriteTimestamp(aCmdBuff, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, aMeasurePool, 0);
		//--------------------------------------------------------------------------------
		//begin second render pass: Vertical blur
		 VkClearValue bloomClearValues[2]{};
		 // Clear to a dark gray background. If we were debugging, this would potentially 
		 // help us see whether the render pass took place, even if nothing else was drawn
		 bloomClearValues[0].color.float32[0] = 0.0f;
		 bloomClearValues[0].color.float32[1] = 0.0f;
		 bloomClearValues[0].color.float32[2] = 0.0f;
		 bloomClearValues[0].color.float32[3] = 1.0f;

		 bloomClearValues[1].depthStencil.depth = 1.0f;

		 VkRenderPassBeginInfo bloomPassInfo{};
		 bloomPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		 bloomPassInfo.renderPass = aBloomPass.renderPass.handle;
		 bloomPassInfo.framebuffer = aBloomPass.framebuffers[0].framebuffer.handle;
		 bloomPassInfo.renderArea.offset = VkOffset2D{ 0,0 };
		 bloomPassInfo.renderArea.extent = VkExtent2D{ aImageExtent.width,aImageExtent.height };
		 bloomPassInfo.clearValueCount = 2;
		 bloomPassInfo.pClearValues = bloomClearValues;

		 vkCmdBeginRenderPass(aCmdBuff, &bloomPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		 vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aVerticalBloomPipe);
		 vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aBloomLayout, 0, 1, &aBloomDescriptors, 0, nullptr);

		 vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

		 //end the render pass
		 vkCmdEndRenderPass(aCmdBuff);
		 //-----------------------------------------------------------------------------------------------------------------------
		 //begin third render pass: Horizontal blur
		 bloomPassInfo.framebuffer = aBloomPass.framebuffers[1].framebuffer.handle;
		 vkCmdBeginRenderPass(aCmdBuff, &bloomPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		 vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aHorizontalBloomPipe);
		 vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aBloomLayout, 0, 1, &aBloomDescriptors2, 0, nullptr);

		 vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

		 //end the render pass
		 vkCmdEndRenderPass(aCmdBuff);
		 //vkCmdWriteTimestamp(aCmdBuff, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, aMeasurePool, 1);
		//------------------------------------------------------------------------------------------------------------------------
		//begin render pass
		VkClearValue clearValues2[2]{};
		// Clear to a dark gray background. If we were debugging, this would potentially 
		// help us see whether the render pass took place, even if nothing else was drawn
		clearValues2[0].color.float32[0] = 0.0f;
		clearValues2[0].color.float32[1] = 0.0f;
		clearValues2[0].color.float32[2] = 0.0f;
		clearValues2[0].color.float32[3] = 1.0f;

		clearValues2[1].depthStencil.depth = 1.0f;

		VkRenderPassBeginInfo passInfo2{};
		passInfo2.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo2.renderPass = aRenderPass;
		passInfo2.framebuffer = aFramebuffer;
		passInfo2.renderArea.offset = VkOffset2D{ 0,0 };
		passInfo2.renderArea.extent = VkExtent2D{ aImageExtent.width,aImageExtent.height };
		passInfo2.clearValueCount = 2;
		passInfo2.pClearValues = clearValues2;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo2, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aPipeline);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aNormalLayout, 0, 1, &aOffscreenDescriptors, 0, nullptr);

		vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

		//end the render pass
		vkCmdEndRenderPass(aCmdBuff);

		//end command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}


	}



	void submit_commands( lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (const auto res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str()
			);
		}
	}

	//void submit_offline_commands(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence,VkSemaphore aSignalSemaphore)
	//{
	//	VkSubmitInfo submitInfo{};
	//	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	//	submitInfo.commandBufferCount = 1;
	//	submitInfo.pCommandBuffers = &aCmdBuff;

	//	submitInfo.waitSemaphoreCount = 0;
	//	submitInfo.pWaitSemaphores = nullptr;
	//	submitInfo.pWaitDstStageMask = nullptr;

	//	submitInfo.signalSemaphoreCount = 1;
	//	submitInfo.pSignalSemaphores = &aSignalSemaphore;

	//	if (const auto res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence);
	//		VK_SUCCESS != res)
	//	{
	//		throw lut::Error("Unable to submit command buffer to queue\n"
	//			"vkQueueSubmit() returned %s", lut::to_string(res).c_str()
	//		);
	//	}
	//}

	void present_results( VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain )
	{
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;

		const auto presentRes = vkQueuePresentKHR(aPresentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			aNeedToRecreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n"
				"vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str()
			);
		}
	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		if(const auto res = vmaCreateImage(aAllocator.allocator,&imageInfo,&allocInfo,&image,& allocation,nullptr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res).c_str()
			);
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0,1,
			0,1
		};

		VkImageView view = VK_NULL_HANDLE;
		if(const auto res = vkCreateImageView(aWindow.device,&viewInfo,nullptr,&view);VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", lut::to_string(res).c_str()
			);
		}

		return { std::move(depthImage),lut::ImageView(aWindow.device,view) };
	}

	void createColorUBOs(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator, std::vector<glsl::ColorUniform> colorUniforms, std::vector<lut::Buffer> &colorUBOs, BakedModel model)
	{
		for (uint32_t i = 0; i < model.meshes.size(); i++) {
			lut::Buffer colorUBO = lut::create_buffer(
				aAllocator,
				sizeof(glsl::ColorUniform),
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VMA_MEMORY_USAGE_GPU_ONLY
			);

			lut::Buffer colorStaging = lut::create_buffer(
				aAllocator,
				sizeof(glsl::ColorUniform),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_MEMORY_USAGE_CPU_TO_GPU
			);

			void* texPtr = nullptr;
			if (const auto res = vmaMapMemory(aAllocator.allocator, colorStaging.allocation, &texPtr);
				VK_SUCCESS != res)
			{
				throw lut::Error("Mapping memory for writing\n"
					"vmaMapMemory() returned %s", lut::to_string(res).c_str()
				);
			}

			std::memcpy(texPtr, &colorUniforms[i], sizeof(glsl::ColorUniform));
			vmaUnmapMemory(aAllocator.allocator, colorStaging.allocation);

			lut::Fence uploadComplete = labutils::create_fence(aWindow);

			// Queue data uploads from staging buffers to the final buffers 
			// This uses a separate command pool for simplicity. 
			lut::CommandPool uploadPool = lut::create_command_pool(aWindow);
			VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aWindow, uploadPool.handle);

			//record copy commands into command buffer
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = 0;
			beginInfo.pInheritanceInfo = nullptr;

			if (const auto res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
				VK_SUCCESS != res)
			{
				throw lut::Error("Beginning command buffer recording\n"
					"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str()
				);
			}
			//texcoords
			VkBufferCopy ccopy{};
			ccopy.size = sizeof(glsl::ColorUniform);

			vkCmdCopyBuffer(uploadCmd, colorStaging.buffer, colorUBO.buffer, 1, &ccopy);

			lut::buffer_barrier(uploadCmd,
				colorUBO.buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
			);

			if (const auto res = vkEndCommandBuffer(uploadCmd);
				VK_SUCCESS != res)
			{
				throw lut::Error("Ending command buffer recording\n"
					"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str()
				);
			}

			//submit transfer commands
			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &uploadCmd;

			if (const auto res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, uploadComplete.handle);
				VK_SUCCESS != res)
			{
				throw lut::Error("Submitting commands\n"
					"vkQueueSubmit() returned %s", lut::to_string(res).c_str()
				);
			}

			if (auto const res = vkWaitForFences(aWindow.device, 1, &uploadComplete.handle,
				VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
			{
				throw lut::Error("Waiting for upload to complete\n"
					"vkWaitForFences() returned %s", lut::to_string(res).c_str()
				);
			}

			colorUBOs.emplace_back(std::move(colorUBO));
		}
	}
	
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
