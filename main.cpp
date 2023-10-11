#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Heat_method_3/Surface_mesh_geodesic_distances_3.h>
#include <CGAL/Surface_mesh_shortest_path.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "learnopengl/shader.h"
#include "learnopengl/camera.h"

#include <iostream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> SurfaceMesh;
typedef CGAL::Polyhedron_3<K> Polyhedron;

typedef boost::graph_traits<SurfaceMesh>::vertex_descriptor VertexDescriptor;
typedef SurfaceMesh::Property_map<VertexDescriptor, double> VertexDistanceMap;
typedef CGAL::Heat_method_3::Surface_mesh_geodesic_distances_3<SurfaceMesh> HeatMethod;

typedef CGAL::Surface_mesh_shortest_path_traits<K, SurfaceMesh> Traits;
typedef CGAL::Surface_mesh_shortest_path<Traits> Surface_mesh_shortest_path;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void window_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
int screenWidth = 800;
int screenHeight = 600;

int frameBufferWidth;
int frameBufferHeight;

// dynamic arrays for geometry
GLfloat *vertices = NULL;
GLuint *edges = NULL;
GLfloat *coarseVertices = NULL;
GLuint *coarseEdges = NULL;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 2.0f));

// mouse press callback
bool mousePressed;
bool firstCursorPositionCallbackOnPress;
glm::vec2 mousePressedPosition;

// rotation
glm::vec2 rotationAngles;
glm::vec2 rotationAnglesDrag;

// user settings
static float phi = 0.125f;
static float prevPhi = 0.0f;

std::vector<SurfaceMesh::Vertex_index> samplePoints(SurfaceMesh* surface) {
    float sumEdgeLengths = 0.0f;

    for (auto edgeIndex : surface->edges()) {
        auto h = surface->halfedge(edgeIndex);
        auto from = surface->point(surface->source(h));
        auto to = surface->point(surface->target(h));
        auto d = CGAL::sqrt(CGAL::squared_distance(from, to));
        sumEdgeLengths += (float) d;
    }

    float averageEdgeLength = sumEdgeLengths / surface->num_edges();
    float r = pow(phi, -1.0f/3.0f) * averageEdgeLength;

    std::vector<SurfaceMesh::Vertex_index> V;
    for (auto vertexIndex : surface->vertices()) {
        V.push_back(vertexIndex);
    }

    std::cout << V.size() << std::endl << std::endl;

    for (auto it = V.begin(); it < V.end(); it++) {
        VertexDistanceMap vertexDistance = surface->add_property_map<VertexDescriptor, double>("v:distance", 0).first;
        VertexDescriptor source = *it;
        HeatMethod hm(*surface);
        hm.add_source(source);
        hm.estimate_geodesic_distances(vertexDistance);

        auto withinRadius = [&vertexDistance, &r](auto v) { return get(vertexDistance, v) < r; };
        V.erase(std::remove_if(it + 1, V.end(), withinRadius), V.end());
    }

    std::cout << V.size() << std::endl << std::endl;
    return V;
}

std::vector<SurfaceMesh::Vertex_index> createNeighbourhoods(SurfaceMesh* surface, std::vector<SurfaceMesh::Vertex_index> sampling) {
    Surface_mesh_shortest_path shortestPaths(*surface);
    shortestPaths.add_source_points(sampling.begin(), sampling.end());

    std::vector<SurfaceMesh::Vertex_index> res(surface->num_vertices());

    for (auto v : surface->vertices()) {
        auto i = v.idx();
        auto loc = *(shortestPaths.shortest_distance_to_source_points(v).second);
        auto f = loc.first;
        auto bc = loc.second;

        auto h = surface->halfedge(f);
        if (bc[0] >= bc[1] && bc[0] >= bc[2]) {
            res.at(i) = surface->source(h);
        } else if (bc[1] >= bc[2]) {
            res.at(i) = surface->target(h);
        } else {
            res.at(i) = surface->target(surface->next(h));
        }
    }

    return res;
}

SurfaceMesh constructCoarserLevel(SurfaceMesh &surface, std::vector<SurfaceMesh::Vertex_index> &sampling, std::vector<SurfaceMesh::Vertex_index> &neighbourhoods) {
    SurfaceMesh res;

    for (auto v : sampling) {
        res.add_vertex(surface.point(v));
    }

    for (auto e : surface.edges()) {
        // if endpoints of e are in different neighbourhoods, then create edge in res between those neighbourhoods
        auto from = surface.source(e.halfedge());
        auto to = surface.target(e.halfedge());

        auto neighbourhoodFrom = neighbourhoods.at(from.idx());
        auto neighbourhoodTo = neighbourhoods.at(to.idx());

        if (neighbourhoodFrom != neighbourhoodTo) {
            auto newFromIdx = std::distance(sampling.begin(), std::find(sampling.begin(), sampling.end(), neighbourhoodFrom));
            auto newToIdx = std::distance(sampling.begin(), std::find(sampling.begin(), sampling.end(), neighbourhoodTo));
            res.add_edge(*(res.vertices_begin() + newFromIdx), *(res.vertices_begin() + newToIdx));
        }
    }

    return res;
}

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "GravoMG Visualisation", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwGetWindowSize(window, &screenWidth, &screenHeight);
    glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);

    // glfw callbacks
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);

    // initialize glew
    // ---------------------------------------
    GLenum glewStatus = glewInit();
    if (glewStatus != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    // build and compile our shader program
    // ------------------------------------
    Shader ourShader("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl");

    // load mesh from file
    // -------------------
    SurfaceMesh surface;
    Polyhedron mesh;

    std::ifstream in("meshes/cactus.off");
    in >> mesh;
    CGAL::copy_face_graph( mesh, surface);

    auto nVertices = surface.num_vertices();
    auto nEdges = surface.num_edges();

    std::cout << "#vertices = " << nVertices << std::endl;
    std::cout << "#edges = " << nEdges << std::endl;

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    vertices = new float[nVertices * 3];
    edges = new uint[nEdges * 2];

    std::vector<glm::vec3> vertexColorsLeft(nVertices);
    std::vector<glm::vec3> vertexColorsRight(nVertices);
    std::vector<glm::vec3> coarseVertexColors;

    auto contains = [](auto v, auto x) { return std::find(v.begin(), v.end(), x) != v.end(); };

    for (auto vertexIndex : surface.vertices()) {
        auto i = vertexIndex.idx();
        auto p = surface.point(vertexIndex);

        vertices[i*3]     = (float) p.x();
        vertices[i*3 + 1] = (float) p.y();
        vertices[i*3 + 2] = (float) p.z();
    }

    for (auto edgeIndex : surface.edges()) {
        auto i = edgeIndex.idx();
        auto h = surface.halfedge(edgeIndex);

        edges[i * 2]     = surface.source(h).idx();
        edges[i * 2 + 1] = surface.target(h).idx();
    }

    unsigned int vertexArray, vertexBuffer, colorBuffer, edgeBuffer, coarseVertexArray, coarseVertexBuffer, coarseColorBuffer, coarseEdgeBuffer;
    glGenVertexArrays(1, &vertexArray);
    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &colorBuffer);
    glGenBuffers(1, &edgeBuffer);
    glGenVertexArrays(1, &coarseVertexArray);
    glGenBuffers(1, &coarseVertexBuffer);
    glGenBuffers(1, &coarseColorBuffer);
    glGenBuffers(1, &coarseEdgeBuffer);

    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, 3 * nVertices * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    // position attribute
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    // color attribute
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * nEdges * sizeof(GLuint), edges, GL_STATIC_DRAW);

    std::vector<CGAL::SM_Vertex_index> sampling;
    std::vector<CGAL::SM_Vertex_index> neighbourhoods;
    SurfaceMesh coarserLevel;

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // render imgui window
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Options", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration);
        ImGui::SetWindowPos(ImVec2(10, 10));
        ImGui::SetWindowSize(ImVec2(120, 40));

        ImGui::DragFloat("phi", &phi, 0.005f);
        if (!ImGui::IsItemActive() && phi != prevPhi) {
            prevPhi = phi;
            sampling = samplePoints(&surface);
            neighbourhoods = createNeighbourhoods(&surface, sampling);
            coarserLevel = constructCoarserLevel(surface, sampling, neighbourhoods);

            std::cout << "#vertices = " << coarserLevel.num_vertices() << std::endl;
            std::cout << "#edges = " << coarserLevel.num_edges() << std::endl;

            for (auto vertexIndex : surface.vertices()) {
                auto i = vertexIndex.idx();
                vertexColorsLeft.at(i) = contains(sampling, vertexIndex) ? glm::vec3(1, 1, 0) : glm::vec3(0);

                int neighbourhood = neighbourhoods.at(i).idx();
                vertexColorsRight.at(i) = glm::vec3(((neighbourhood * 3) % 255) / 255.0f, ((neighbourhood * 5) % 255) / 255.0f, ((neighbourhood * 7) % 255) / 255.0f);
            }

            // set up vertex data (and buffer(s)) and configure vertex attributes
            // ------------------------------------------------------------------
            coarseVertices = new float[coarserLevel.num_vertices() * 3];
            coarseEdges = new uint[coarserLevel.num_edges() * 2];

            coarseVertexColors = std::vector<glm::vec3>(coarserLevel.num_vertices());

            for (auto vertexIndex : coarserLevel.vertices()) {
                auto i = vertexIndex.idx();
                auto p = coarserLevel.point(vertexIndex);

                coarseVertices[i*3]     = (float) p.x();
                coarseVertices[i*3 + 1] = (float) p.y();
                coarseVertices[i*3 + 2] = (float) p.z();

                coarseVertexColors.at(i) = glm::vec3((((int)pow(i, 3)) % 255) / 255.0f, (((int)pow(i, 4)) % 255) / 255.0f, (((int)pow(i, 5)) % 255) / 255.0f);
            }

            for (auto edgeIndex : coarserLevel.edges()) {
                auto i = edgeIndex.idx();
                auto h = coarserLevel.halfedge(edgeIndex);

                coarseEdges[i * 2]     = coarserLevel.source(h).idx();
                coarseEdges[i * 2 + 1] = coarserLevel.target(h).idx();
            }

            // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
            glBindVertexArray(coarseVertexArray);

            glBindBuffer(GL_ARRAY_BUFFER, coarseVertexBuffer);
            glBufferData(GL_ARRAY_BUFFER, 3 * coarserLevel.num_vertices() * sizeof(GLfloat), coarseVertices, GL_STATIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, coarseColorBuffer);
            glBufferData(GL_ARRAY_BUFFER, coarseVertexColors.size() * sizeof(glm::vec3), &coarseVertexColors[0], GL_STATIC_DRAW);

            // position attribute
            glBindBuffer(GL_ARRAY_BUFFER, coarseVertexBuffer);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(0);

            // color attribute
            glBindBuffer(GL_ARRAY_BUFFER, coarseColorBuffer);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(1);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseEdgeBuffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * coarserLevel.num_edges() * sizeof(GLuint), coarseEdges, GL_STATIC_DRAW);
        }

        ImGui::End();

        // input
        // -----
        processInput(window);

        int vpWidth = 0.333 * frameBufferWidth;
        int vpHeight = 0.9 * frameBufferHeight;

        // pass projection matrix to shader (note that in this case it could change every frame)
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)vpWidth / (float)vpHeight, 0.1f, 100.0f);
        ourShader.setMat4("projection", projection);

        // camera/view transformation
        glm::mat4 view = camera.GetViewMatrix();
        ourShader.setMat4("view", view);

        // calculate the model matrix and pass it to shader before drawing
        glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        model = glm::rotate(model, rotationAngles.x + rotationAnglesDrag.x, glm::vec3(1, 0, 0));
        model = glm::rotate(model, rotationAngles.y + rotationAnglesDrag.y, glm::vec3(0, 1, 0));
        ourShader.setMat4("model", model);

        // set point size
        glPointSize(5 * 45 / camera.Zoom);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, vpWidth, vpHeight);

        // render the triangle
        ourShader.use();

        glBindVertexArray(vertexArray);

        glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
        glBufferData(GL_ARRAY_BUFFER, vertexColorsLeft.size() * sizeof(glm::vec3), &vertexColorsLeft[0], GL_STATIC_DRAW);

        // draw edges
        glDisableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
        glDrawElements(GL_LINES, 2 * nEdges, GL_UNSIGNED_INT, 0);

        // draw vertices
        glEnableVertexAttribArray(1);
        glBindVertexArray(vertexArray);
        glDrawArrays(GL_POINTS, 0, nVertices);

        glViewport(vpWidth, 0, vpWidth, vpHeight);

        // render the triangle
        ourShader.use();

        glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
        glBufferData(GL_ARRAY_BUFFER, vertexColorsRight.size() * sizeof(glm::vec3), &vertexColorsRight[0], GL_STATIC_DRAW);

        // draw edges
        glDisableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
        glDrawElements(GL_LINES, 2 * nEdges, GL_UNSIGNED_INT, 0);

        // draw vertices
        glEnableVertexAttribArray(1);
        glBindVertexArray(vertexArray);
        glDrawArrays(GL_POINTS, 0, nVertices);

        glViewport(2 * vpWidth, 0, vpWidth, vpHeight);

        // render the triangle
        ourShader.use();

        glBindVertexArray(coarseVertexArray);

        // draw edges
        glDisableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseEdgeBuffer);
        glDrawElements(GL_LINES, 2 * coarserLevel.num_edges(), GL_UNSIGNED_INT, 0);

        // draw vertices
        glEnableVertexAttribArray(1);
        glBindVertexArray(coarseVertexArray);
        glDrawArrays(GL_POINTS, 0, coarserLevel.num_vertices());

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &vertexArray);
    glDeleteBuffers(1, &vertexBuffer);
    glDeleteBuffers(1, &colorBuffer);
    glDeleteBuffers(1, &edgeBuffer);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    frameBufferWidth = width;
    frameBufferHeight = height;
}

// glfw: whenever a mouse button is pressed, this callback is  called
// ------------------------------------------------------------------
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            firstCursorPositionCallbackOnPress = true;
        } else {
            mousePressed = false;
            firstCursorPositionCallbackOnPress = false;

            rotationAngles += rotationAnglesDrag;
            rotationAnglesDrag = glm::vec2(0);
        }
    }
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    if (mousePressed) {
        if (firstCursorPositionCallbackOnPress) {
            mousePressedPosition = glm::vec2(xpos, ypos);
            firstCursorPositionCallbackOnPress = false;
        }

        float xRotationAngleDrag = (ypos - mousePressedPosition.y) * 2 * M_PI / screenHeight;
        float yRotationAngleDrag = (xpos - mousePressedPosition.x) * 2 * M_PI / screenWidth;
        rotationAnglesDrag = glm::vec2(xRotationAngleDrag, yRotationAngleDrag);
    }
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

// glfw: whenever the window is resized, this callback is fired
void window_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
}
