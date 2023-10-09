#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polyhedron_3.h>

#include "learnopengl/shader.h"
#include "learnopengl/camera.h"

#include <iostream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> SurfaceMesh;
typedef CGAL::Polyhedron_3<K> Polyhedron;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// dynamic arrays for geometry
GLfloat *vertices = NULL;
GLuint *edges = NULL;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 2.0f));

// mouse press callback
bool mousePressed;
bool firstCursorPositionCallbackOnPress;
glm::vec2 mousePressedPosition;

// rotation
glm::vec2 rotationAngles;
glm::vec2 rotationAnglesDrag;

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "GravoMG Visualisation", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // glfw callbacks
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // initialize glew
    // ---------------------------------------
    GLenum glewStatus = glewInit();
    if (glewStatus != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

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

    unsigned int vertexArray, vertexBuffer, colorBuffer, edgeBuffer;
    glGenVertexArrays(1, &vertexArray);
    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &colorBuffer);
    glGenBuffers(1, &edgeBuffer);

    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, 3 * nVertices * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

//    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
//    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);

    // position attribute
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    // color attribute
//    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
//    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
//    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * nEdges * sizeof(GLuint), edges, GL_STATIC_DRAW);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // pass projection matrix to shader (note that in this case it could change every frame)
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
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
        glPointSize(2 * 45 / camera.Zoom);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // render the triangle
        ourShader.use();

        // draw vertices
        glBindVertexArray(vertexArray);
        glDrawArrays(GL_POINTS, 0, nVertices);

        // draw edges
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
        glDrawElements(GL_LINES, 2 * nEdges, GL_UNSIGNED_INT, 0);

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
    glViewport(0, 0, width, height);
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
    if (mousePressed) {
        if (firstCursorPositionCallbackOnPress) {
            mousePressedPosition = glm::vec2(xpos, ypos);
            firstCursorPositionCallbackOnPress = false;
        }

        float xRotationAngleDrag = (ypos - mousePressedPosition.y) * M_PI / SCR_HEIGHT;
        float yRotationAngleDrag = (xpos - mousePressedPosition.x) * M_PI / SCR_WIDTH;
        rotationAnglesDrag = glm::vec2(xRotationAngleDrag, yRotationAngleDrag);
    }
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
