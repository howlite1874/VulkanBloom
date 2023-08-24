#pragma once

#include <volk/volk.h>
#include <vk_mem_alloc.h>

#include <utility>

#include <cassert>

#include "vulkan_context.hpp"

#include <unordered_map>
#include <mutex>

#ifndef VMA_CALL_CONV
#define VMA_CALL_CONV
#endif

namespace labutils
{

	class Allocator
	{
		public:
			Allocator() noexcept, ~Allocator();

			explicit Allocator( VmaAllocator ) noexcept;

			Allocator( Allocator const& ) = delete;
			Allocator& operator= (Allocator const&) = delete;

			Allocator( Allocator&& ) noexcept;
			Allocator& operator = (Allocator&&) noexcept;

		public:
			VmaAllocator allocator = VK_NULL_HANDLE;
	};

	Allocator create_allocator( VulkanContext const& );

}
