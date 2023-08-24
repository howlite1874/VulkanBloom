#pragma once

#include <cstdint>
#include <vector> 

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp"
#include "../cw3/baked_model.hpp"

struct objMesh
{
	labutils::Buffer positions;
	labutils::Buffer texcoords;
	labutils::Buffer normals;
	labutils::Buffer indices;


	objMesh() = default;

	objMesh(objMesh&& other) noexcept :
		positions(std::move(other.positions)),
		texcoords(std::move(other.texcoords)),
		normals(std::move(other.normals)),
		indices(std::move(other.indices)
	)
{}
};


objMesh create_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, BakedMeshData aModel);

