#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include <stb_image.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <array>
#include <set>
#include <sstream>
#include "dbgstream.h"
#include <vulkan.hpp>

const int WIDTH = 800;
const int HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


static  VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
	auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pCallback);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

static void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
	if (func != nullptr) {
		func(instance, callback, pAllocator);
	}
}

struct QueueFamilyIndices {
    int graphicsFamily = -1;
    int presentFamily = -1;
	constexpr QueueFamilyIndices() : graphicsFamily(-1), presentFamily(-1) {}
	QueueFamilyIndices(vk::PhysicalDevice device, vk::SurfaceKHR surface) : QueueFamilyIndices() {
		auto queueFamilies = device.getQueueFamilyProperties();

		for (int i = 0; i < queueFamilies.size() && !isComplete(); i++) {
			const auto& queueFamily = queueFamilies[i];
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
				graphicsFamily = i;
			}
			if (queueFamily.queueCount > 0 && device.getSurfaceSupportKHR(i, surface)) {
				presentFamily = i;
			}
		}
	}

    constexpr bool isComplete() const {
        return graphicsFamily >= 0 && presentFamily >= 0;
    }
};

struct SwapChainSupportDetails {
	vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
	SwapChainSupportDetails() = default;
	SwapChainSupportDetails(vk::PhysicalDevice  device, vk::SurfaceKHR surface) :
		capabilities(device.getSurfaceCapabilitiesKHR(surface)),
		formats(device.getSurfaceFormatsKHR(surface)),
		presentModes(device.getSurfacePresentModesKHR(surface))
	{}
};

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
		return  vk::VertexInputBindingDescription();
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions = {
			vk::VertexInputAttributeDescription(0,0,vk::Format::eR32G32Sfloat,offsetof(Vertex, pos)),
			vk::VertexInputAttributeDescription(0,1,vk::Format::eR32G32B32Sfloat,offsetof(Vertex, color)),
			vk::VertexInputAttributeDescription(0,2,vk::Format::eR32G32Sfloat,offsetof(Vertex, texCoord)),
		};
        return attributeDescriptions;
    }
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

static const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
};

static const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

	vk::Instance instance;
	vk::DebugReportCallbackEXT callback;
	vk::SurfaceKHR surface;
	
	vk::PhysicalDevice physicalDevice;
    vk::Device device;
	
	vk::Queue graphicsQueue;
	vk::Queue presentQueue;

	vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;

    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::ImageView> swapChainImageViews;
    std::vector<vk::Framebuffer> swapChainFramebuffers;

    vk::RenderPass renderPass;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;

    vk::CommandPool commandPool;

    vk::Image textureImage;
    vk::DeviceMemory textureImageMemory;
    vk::ImageView textureImageView;
    vk::Sampler textureSampler;

    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;

    vk::Buffer uniformBuffer;
    vk::DeviceMemory uniformBufferMemory;

    vk::DescriptorPool descriptorPool;
    vk::DescriptorSet descriptorSet;

    std::vector<vk::CommandBuffer> commandBuffers;

    vk::Semaphore imageAvailableSemaphore;
    vk::Semaphore renderFinishedSemaphore;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

        glfwSetWindowUserPointer(window, this);
        glfwSetWindowSizeCallback(window, HelloTriangleApplication::onWindowResized);
    }

    void initVulkan() {
        createInstance();
        setupDebugCallback();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffer();
        createDescriptorPool();
        createDescriptorSet();
        createCommandBuffers();
        createSemaphores();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            updateUniformBuffer();
            drawFrame();
        }
		device.waitIdle();
    }

    void cleanupSwapChain() {
		for (auto& it : swapChainFramebuffers) {
			device.destroyFramebuffer(it);
			it = nullptr;
		}
		device.freeCommandBuffers(commandPool, commandBuffers);
		commandPool = nullptr;

		device.destroyPipeline(graphicsPipeline);
		graphicsPipeline = nullptr;
		device.destroyPipelineLayout(pipelineLayout);
		pipelineLayout = nullptr;
		device.destroyRenderPass(renderPass);
		renderPass = nullptr;

		for (auto& it : swapChainImageViews) {
			device.destroyImageView(it);
			it = nullptr;
		}

		device.destroySwapchainKHR(swapChain);
		swapChain = nullptr;
    }

    void cleanup() {
        cleanupSwapChain();
		
        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);

        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyBuffer(device, uniformBuffer, nullptr);
        vkFreeMemory(device, uniformBufferMemory, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);
		//instance.destroyDebugReportCallbackEXT(callback); callback = nullptr;

		DestroyDebugReportCallbackEXT(static_cast<VkInstance>(instance), callback, nullptr);

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    static void onWindowResized(GLFWwindow* window, int width, int height) {
        if (width == 0 || height == 0) return;

        HelloTriangleApplication* app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->recreateSwapChain();
    }

    void recreateSwapChain() {
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        vk::ApplicationInfo appInfo;
                appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        vk::InstanceCreateInfo createInfo;
                createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }
		instance = vk::createInstance(createInfo);

    }

    void setupDebugCallback() {
		if (!enableValidationLayers) return;

		VkDebugReportCallbackCreateInfoEXT createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
		createInfo.pfnCallback = debugCallback;
		VkDebugReportCallbackEXT temp;
		if (CreateDebugReportCallbackEXT(static_cast<VkInstance>(instance), &createInfo, nullptr, &temp) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug callback!");
		}
		callback = temp;
    }

    void createSurface() {
		VkSurfaceKHR pSurface;
        if (glfwCreateWindowSurface(instance, window, nullptr, &pSurface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
		surface = pSurface;
    }

    void pickPhysicalDevice() {
		auto devices = instance.enumeratePhysicalDevices();
		

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (!physicalDevice) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<int> uniqueQueueFamilies = {indices.graphicsFamily, indices.presentFamily};

        float queuePriority = 1.0f;
        for (int queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo;
                        queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        vk::PhysicalDeviceFeatures deviceFeatures;
        deviceFeatures.samplerAnisotropy = 1U;

        vk::DeviceCreateInfo createInfo;
        
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }
		device = physicalDevice.createDevice(createInfo);

		graphicsQueue = device.getQueue(indices.graphicsFamily, 0);
		presentQueue = device.getQueue(indices.presentFamily, 0);
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

		vk::SwapchainCreateInfoKHR  createInfo;

        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {(uint32_t) indices.graphicsFamily, (uint32_t) indices.presentFamily};

        if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
			createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } 

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
		createInfo.clipped = 1U;
		swapChain = device.createSwapchainKHR(createInfo);

		swapChainImages = device.getSwapchainImagesKHR(swapChain);

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat);
        }
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment;
        colorAttachment.format = swapChainImageFormat;
		
        colorAttachment.samples = vk::SampleCountFlagBits::e1;;
		colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
		colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;
	
		vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);


        vk::SubpassDescription subpass;
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        vk::SubpassDependency dependency;

        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
       // dependency.dstSubpass = 0;
        dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
       // dependency.srcAccessMask = 0;
        dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;

        vk::RenderPassCreateInfo renderPassInfo;
                renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
		renderPass = device.createRenderPass(renderPassInfo);
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding;
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
		
        uboLayoutBinding.pImmutableSamplers = nullptr;
        uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
		
        vk::DescriptorSetLayoutBinding samplerLayoutBinding;
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
        vk::DescriptorSetLayoutCreateInfo layoutInfo;
                layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
		descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("shader_textures_vert.spv");
        auto fragShaderCode = readFile("shader_textures_frag.spv");
		
        vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo(
			vk::PipelineShaderStageCreateFlags(),
			vk::ShaderStageFlagBits::eVertex,
			vertShaderModule,
			"main"
		);
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo(
			vk::PipelineShaderStageCreateFlags(),
			vk::ShaderStageFlagBits::eFragment,
			fragShaderModule,
			"main"
		);

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
        

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
        
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
        inputAssembly.primitiveRestartEnable = 0U;

		vk::Viewport viewport(0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);

		vk::Rect2D scissor({ 0, 0 }, swapChainExtent);

		vk::PipelineViewportStateCreateInfo viewportState(
			vk::PipelineViewportStateCreateFlags(),
			1, &viewport,
			1, &scissor
		);

		vk::PipelineRasterizationStateCreateInfo rasterizer(
			vk::PipelineRasterizationStateCreateFlags(),
			0U, // depthClampEnable
			0U, // rasterizerDiscardEnable
			vk::PolygonMode::eFill,
			vk::CullModeFlagBits::eBack,
			vk::FrontFace::eCounterClockwise,
			0U, // depthBiasEnable
			1.0f, // depthBiasConstantFactor_
			1.0f, // depthBiasClamp_
			1.0f, // depthBiasSlopeFactor_
			1.0f // lineWidth_
		);


		vk::PipelineMultisampleStateCreateInfo multisampling(
			vk::PipelineMultisampleStateCreateFlags(), 
			vk::SampleCountFlagBits::e1, 
			0U // sample shading enable
		);

        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
		
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = 0U;

        vk::PipelineColorBlendStateCreateInfo colorBlending;
      
        colorBlending.logicOpEnable = 0U;
        colorBlending.logicOp = vk::LogicOp::eCopy;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo;
        
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = nullptr;
		graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo);

		device.destroyShaderModule(fragShaderModule);
		device.destroyShaderModule(vertShaderModule);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            vk::ImageView attachments[] = {
                swapChainImageViews[i]
            };

            vk::FramebufferCreateInfo framebufferInfo;
            
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;
			swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        vk::CommandPoolCreateInfo poolInfo;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

		commandPool = device.createCommandPool(poolInfo);
    }

	void createTextureImage() {
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load("images/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		vk::DeviceSize imageSize = texWidth * texHeight * 4;

		if (!pixels) {
			throw std::runtime_error("failed to load texture image!");
		}

		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;

		createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		{
			void * data = device.mapMemory(stagingBufferMemory, 0, imageSize);
			::memcpy(data, pixels, imageSize);
			device.unmapMemory(stagingBufferMemory);
 			//std::copy(pixels, pixels + element_count, data.begin());
		}


		// vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		 //    memcpy(data, pixels, static_cast<size_t>(imageSize));
	   //  vkUnmapMemory(device, stagingBufferMemory);

		stbi_image_free(pixels);
		
        createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, 
			vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);
	
        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
	
        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

		device.destroyBuffer(stagingBuffer, nullptr);
		device.freeMemory(stagingBufferMemory, nullptr);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Unorm);
    }

    void createTextureSampler() {
        vk::SamplerCreateInfo samplerInfo;
		
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        samplerInfo.anisotropyEnable = 1U;
        samplerInfo.maxAnisotropy = 16;
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = 0U;
        samplerInfo.compareEnable = 0U;
        samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		textureSampler = device.createSampler(samplerInfo);
    }

    vk::ImageView createImageView(vk::Image image, vk::Format format) {
		vk::ImageViewCreateInfo viewInfo;
        viewInfo.image = image; 
		viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

        return device.createImageView(viewInfo);
    }

    void  createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory) {
        vk::ImageCreateInfo imageInfo;
        
        imageInfo.imageType = vk::ImageType::e2D; 
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = vk::ImageLayout::eUndefined;
        imageInfo.usage = usage;
		imageInfo.samples = vk::SampleCountFlagBits::e1;
		
		imageInfo.sharingMode = vk::SharingMode::eExclusive;

		image = device.createImage(imageInfo, nullptr);

		vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);
		
		vk::MemoryAllocateInfo allocInfo(memRequirements.size, findMemoryType(memRequirements.memoryTypeBits, properties));
        
		imageMemory = device.allocateMemory(allocInfo, nullptr);

		device.bindImageMemory(image, imageMemory, 0);
    }

    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier;
        
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
		
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
		
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;
	
        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
			barrier.srcAccessMask = vk::AccessFlags(); // static_cast<vk::AccessFlagBits>(0U);
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
			
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
			
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
			
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
		commandBuffer.pipelineBarrier(
            sourceStage, destinationStage,
			vk::DependencyFlags(),
			nullptr,
			nullptr,
            barrier
        );
        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferImageCopy region;
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0; 
        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {
            width,
            height,
            1
        };
		commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);

        endSingleTimeCommands(commandBuffer);
    }

    void createVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
		
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);


		
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
		
		device.destroyBuffer(stagingBuffer, nullptr);
	
		device.freeMemory(stagingBufferMemory, nullptr);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
		std::memcpy(data, vertices.data(), (size_t)bufferSize);
		device.unmapMemory(stagingBufferMemory);

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		device.destroyBuffer(stagingBuffer, nullptr);

		device.freeMemory(stagingBufferMemory, nullptr);
    }

    void createUniformBuffer() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffer, uniformBufferMemory);
    }

    void createDescriptorPool() {
		std::array<vk::DescriptorPoolSize, 2> poolSizes = {
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,1),
			vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler,1),
		};

		vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlags(),1, static_cast<uint32_t>(poolSizes.size()), poolSizes.data());

		descriptorPool = device.createDescriptorPool(poolInfo, nullptr);
    }

    void createDescriptorSet() {
		vk::DescriptorSetLayout layouts[] = { descriptorSetLayout };
		vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1, layouts);

		descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		vk::DescriptorBufferInfo bufferInfo(uniformBuffer, 0, sizeof(UniformBufferObject));
		vk::DescriptorImageInfo imageInfo(textureSampler, textureImageView, vk::ImageLayout::eShaderReadOnlyOptimal);
		std::array<vk::WriteDescriptorSet, 2> descriptorWrites = {
			vk::WriteDescriptorSet(descriptorSet, 0U, 0U,1, vk::DescriptorType::eUniformBuffer,nullptr, &bufferInfo),
			vk::WriteDescriptorSet(descriptorSet, 1U, 0U,1, vk::DescriptorType::eCombinedImageSampler,&imageInfo,nullptr),
		};

		device.updateDescriptorSets(descriptorWrites, nullptr);
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
        vk::BufferCreateInfo bufferInfo;
                bufferInfo.size = size;
        bufferInfo.usage = usage;
            bufferInfo.sharingMode = vk::SharingMode::eExclusive;
		buffer = device.createBuffer(bufferInfo);

		vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
		

        vk::MemoryAllocateInfo allocInfo;
                allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		bufferMemory = device.allocateMemory(allocInfo);
		device.bindBufferMemory(buffer, bufferMemory, 0);
    }

    vk::CommandBuffer beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo;
                allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        vk::CommandBuffer commandBuffer;

		commandBuffer = device.allocateCommandBuffers(allocInfo)[0];
		
        vk::CommandBufferBeginInfo beginInfo;
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

		vk::SubmitInfo submitInfo;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
		graphicsQueue.submit(submitInfo,nullptr);
		graphicsQueue.waitIdle();
        //vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		device.freeCommandBuffers(commandPool, commandBuffer);
    }

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

		vk::BufferCopy copyRegion(size);

		commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);
        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
	

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        vk::CommandBufferAllocateInfo allocInfo;
        allocInfo.commandPool = commandPool; 
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = (uint32_t)swapChainFramebuffers.size();
		commandBuffers = device.allocateCommandBuffers(allocInfo);

			for(size_t i=0; i < commandBuffers.size(); i++) {
				auto& cmdBuffer = commandBuffers[i];
			vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
				cmdBuffer.begin(beginInfo);
				vk::ClearValue clearColor;
				vk::RenderPassBeginInfo renderPassInfo(renderPass, swapChainFramebuffers[i], vk::Rect2D(vk::Offset2D(), swapChainExtent), 1, &clearColor);
				cmdBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
				cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

                vk::Buffer vertexBuffers[] = {vertexBuffer};
                vk::DeviceSize offsets[] = {0};
				cmdBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

				cmdBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
				cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

				cmdBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

				cmdBuffer.endRenderPass();
				cmdBuffer.end();
        }
    }

    void createSemaphores() {
        vk::SemaphoreCreateInfo semaphoreInfo;
		imageAvailableSemaphore  = device.createSemaphore(semaphoreInfo);
		renderFinishedSemaphore  = device.createSemaphore(semaphoreInfo);
    }

    void updateUniformBuffer() {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0f;

		UniformBufferObject ubo;
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

		void * data = device.mapMemory(uniformBufferMemory, 0, sizeof(ubo), vk::MemoryMapFlags());
		std::memcpy(data, &ubo, sizeof(ubo));
		device.unmapMemory(uniformBufferMemory);
    }

    void drawFrame() {

		auto result = device.acquireNextImageKHR(swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore, nullptr);
		switch (result.result) {
		case vk::Result::eErrorOutOfDateKHR:
			recreateSwapChain();
			return;
		case vk::Result::eSuboptimalKHR:
		case vk::Result::eSuccess:
			break;
		default:
			throw std::runtime_error("failed to acquire swap chain image!");
		};
		uint32_t imageIndex = result.value;
		vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
		vk::Semaphore waitSemaphores[] = { imageAvailableSemaphore };
		vk::Semaphore signalSemaphores[] = { renderFinishedSemaphore };
		vk::SubmitInfo submitInfo(waitSemaphores, { commandBuffers[imageIndex] }, signalSemaphores, waitStages);

		graphicsQueue.submit(submitInfo, nullptr);
		vk::SwapchainKHR swapChains[] = { swapChain };
		vk::PresentInfoKHR presentInfo(signalSemaphores, swapChains, &imageIndex);
        

		switch (presentQueue.presentKHR(presentInfo)){
		case vk::Result::eErrorOutOfDateKHR:
		case vk::Result::eSuboptimalKHR:
			recreateSwapChain();
			break;
		case vk::Result::eSuccess:
			break;
		default:
			throw std::runtime_error("failed to present swap chain image!");
		};
		presentQueue.waitIdle();

    }
	
	vk::ShaderModule createShaderModule(const std::vector<char>& code) {
		vk::ShaderModuleCreateInfo createInfo(code.size(), reinterpret_cast<const uint32_t*>(code.data()));
		auto shaderModule = device.createShaderModule(createInfo);
        return shaderModule;
    }
	
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) { 
		// what does this even do?
		static constexpr vk::SurfaceFormatKHR default_format = { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };

		if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined)  return default_format;

        for (const auto& availableFormat : availableFormats) {
            if (default_format == availableFormat)    return availableFormat;
        }

        return availableFormats[0];
    }
	
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
		vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            } else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                bestMode = availablePresentMode;
            }
        }

        return bestMode;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetWindowSize(window, &width, &height);

			vk::Extent2D actualExtent(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
		//	actualExtent = std::clamp(actualExtent, capabilities.minImageExtent, capabilities.maxImageExtent.width);
			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device) {
		SwapChainSupportDetails details;
		details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
		details.formats = device.getSurfaceFormatsKHR(surface);
		details.presentModes = device.getSurfacePresentModesKHR(surface);
		

        return details;
    }

    bool isDeviceSuitable(vk::PhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);
		
        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

		vk::PhysicalDeviceFeatures supportedFeatures = device.getFeatures();
		

        return indices.isComplete() && extensionsSupported && supportedFeatures.samplerAnisotropy;
    }

    bool checkDeviceExtensionSupport(vk::PhysicalDevice device) {
		std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties();
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) {
        QueueFamilyIndices indices;
		auto queueFamilies = device.getQueueFamilyProperties();

		for(int i =0; i < queueFamilies.size() && !indices.isComplete(); i++){
			const auto& queueFamily = queueFamilies[i];
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }
            if (queueFamily.queueCount > 0 && device.getSurfaceSupportKHR(i,surface)) {
                indices.presentFamily = i;
            }
        }
        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        std::vector<const char*> extensions;

        unsigned int glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        for (unsigned int i = 0; i < glfwExtensionCount; i++) {
            extensions.push_back(glfwExtensions[i]);
        }

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
		auto availableLayers = vk::enumerateInstanceLayerProperties();

        for (const char* layerName : validationLayers) {
            bool layerFound = false;
		
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }


    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, int32_t code, const char* layerPrefix, const char* msg, void* userData) {
        std::cerr << "validation layer: " << msg << std::endl;

        return VK_FALSE;
    }
};


int new_texture_mapping() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::runtime_error& e) {

		dbg << e.what() << std::endl;

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
