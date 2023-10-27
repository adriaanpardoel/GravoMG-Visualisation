#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polyhedron_3.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "learnopengl/shader.h"
#include "learnopengl/camera.h"

#include "gravomg/multigrid_solver.h"

#include <iostream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> SurfaceMesh;
typedef CGAL::Polyhedron_3<K> Polyhedron;

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
GLuint *candidateEdges = NULL;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 2.0f));

// mouse press callback
bool mousePressed;
bool firstCursorPositionCallbackOnPress;
glm::vec2 mousePressedPosition;
glm::vec2 currentMousePosition;

// rotation
glm::vec2 rotationAngles;
glm::vec2 rotationAnglesDrag;

// user settings
static float phi = 0.125f;
static float prevPhi = 0.0f;

CGAL::SM_Vertex_index selectedPoint = CGAL::SM_Vertex_index(0);
CGAL::SM_Vertex_index prevSelectedPoint = CGAL::SM_Vertex_index(0);

bool useBarycentricCoords;
bool useEdgeCoords;
bool useInvDistWeights;

Eigen::RowVectorXd::Index coarsePoint;


glm::mat4 model, view, projection;
SurfaceMesh surface;
glm::vec3 rayCamPos;


std::vector<Eigen::MatrixXd> hierarchyVertices;
std::vector<std::vector<std::vector<int>>> hierarchyTriangles;
std::vector<Eigen::MatrixXi> hierarchyNeighbours;
std::vector<std::vector<int>> hierarchySampling;
std::vector<Eigen::SparseMatrix<double>> hierarchyProlongation;
std::vector<std::vector<bool>> hierarchyProlongationFallback;


std::vector<std::vector<int>> candidateTri;
int nCoarseEdges;
std::vector<int> prolongationVertices;
std::vector<double> prolongationWeights;
int edgeCoordEdgeIndex;


glm::vec3 toVec3(CGAL::Point_3<CGAL::Epick> p) {
    return {p.x(), p.y(), p.z()};
}

void constructHierarchy(SurfaceMesh &surface) {
    Eigen::MatrixXd positions(surface.num_vertices(), 3);
    Eigen::MatrixXi neighbours(surface.num_vertices(), surface.num_vertices());
    Eigen::SparseMatrix<double> mass(surface.num_vertices(), surface.num_vertices());

    neighbours.setConstant(-1);
    int maxNeighbours = 0;

    for (auto v : surface.vertices()) {
        auto p = surface.point(v);

        positions(v.idx(), 0) = p.x();
        positions(v.idx(), 1) = p.y();
        positions(v.idx(), 2) = p.z();

        int col = 0;
        for (auto outEdge : CGAL::halfedges_around_source(v, surface)) {
            auto neighbour = surface.target(outEdge);
            neighbours(v.idx(), col++) = neighbour.idx();
        }

        maxNeighbours = std::max(maxNeighbours, col);
    }

    neighbours.conservativeResize(neighbours.rows(), maxNeighbours);

    mass.setIdentity();

    GravoMG::MultigridSolver solver(positions, neighbours, mass);
    solver.debug = true;
    solver.lowBound = 0;
    solver.ratio = 1.0f / phi;
    solver.buildHierarchy();

    hierarchyVertices = solver.levelV;
    hierarchyTriangles = solver.allTriangles;
    hierarchyNeighbours = solver.neighHierarchy;
    hierarchySampling = solver.samples;
    hierarchyProlongation = solver.U;
    hierarchyProlongationFallback = solver.prolongationFallback;
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

    unsigned int vertexArray, vertexBuffer, colorBuffer, edgeBuffer, coarseVertexArray, coarseVertexBuffer, coarseColorBuffer, coarseEdgeBuffer, coarseCandidateEdgeBuffer, finalTriangleBuffer;
    glGenVertexArrays(1, &vertexArray);
    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &colorBuffer);
    glGenBuffers(1, &edgeBuffer);
    glGenVertexArrays(1, &coarseVertexArray);
    glGenBuffers(1, &coarseVertexBuffer);
    glGenBuffers(1, &coarseColorBuffer);
    glGenBuffers(1, &coarseEdgeBuffer);
    glGenBuffers(1, &coarseCandidateEdgeBuffer);
    glGenBuffers(1, &finalTriangleBuffer);

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
        if (!ImGui::IsItemActive() && (phi != prevPhi || selectedPoint != prevSelectedPoint)) {
            prevPhi = phi;
            prevSelectedPoint = selectedPoint;

            constructHierarchy(surface);

            Eigen::VectorXi neighbourhoods(hierarchyProlongation[0].rows());
            for (int v = 0; v < hierarchyProlongation[0].rows(); v++) {
                Eigen::RowVectorXd weightsRow = hierarchyProlongation[0].row(v);
                Eigen::RowVectorXd::Index maxIndex;
                weightsRow.maxCoeff(&maxIndex);
                neighbourhoods[v] = maxIndex;
            }

            coarsePoint = neighbourhoods[selectedPoint];

            for (auto vertexIndex : surface.vertices()) {
                auto i = vertexIndex.idx();
                vertexColorsLeft.at(i) = contains(hierarchySampling[0], vertexIndex.idx()) ? glm::vec3(1, 1, 0) : glm::vec3(0);

                if (i == selectedPoint) {
                    vertexColorsLeft.at(i) = glm::vec3(1.0f, 0.06f, 0.94f);
                }

                int neighbourhood = neighbourhoods[i];
                vertexColorsRight.at(i) = glm::vec3(((neighbourhood * 3) % 255) / 255.0f, ((neighbourhood * 5) % 255) / 255.0f, ((neighbourhood * 7) % 255) / 255.0f);
            }

            nCoarseEdges = ((hierarchyNeighbours[0].array() >= 0).cast<int>().sum()) / 2;

            candidateTri.clear();
            std::copy_if(hierarchyTriangles[0].begin(), hierarchyTriangles[0].end(), std::back_inserter(candidateTri), [&contains](auto t) { return contains(t, coarsePoint); });

            // set up vertex data (and buffer(s)) and configure vertex attributes
            // ------------------------------------------------------------------
            coarseVertices = new float[(hierarchyVertices[0].rows() + 1) * 3];
            coarseEdges = new uint[(nCoarseEdges + 3) * 2];
            candidateEdges = new uint[candidateTri.size() * 6];

            coarseVertexColors = std::vector<glm::vec3>(hierarchyVertices[0].rows());

            for (int i = 0; i < hierarchyVertices[0].rows(); i++) {
                auto p = hierarchyVertices[0].row(i);

                coarseVertices[i*3]     = (float) p.x();
                coarseVertices[i*3 + 1] = (float) p.y();
                coarseVertices[i*3 + 2] = (float) p.z();
            }

            prolongationVertices.clear();
            prolongationWeights.clear();

            Eigen::RowVectorXd weightsRow = hierarchyProlongation[0].row(selectedPoint);

            if (hierarchyProlongationFallback[0][selectedPoint]) {
                useBarycentricCoords = false;
                useEdgeCoords = false;
                useInvDistWeights = true;
            } else {
                int nElements = (weightsRow.array() > 0.0).cast<int>().sum();
                useBarycentricCoords = nElements == 3;
                useEdgeCoords = !useBarycentricCoords;
            }

            for (int j = 0; j < hierarchyProlongation[0].cols(); j++) {
                if (weightsRow[j] > 0) {
                    prolongationVertices.push_back(j);
                    prolongationWeights.push_back(weightsRow[j]);
                }
            }

            if (useBarycentricCoords || useEdgeCoords) {
                auto proj = weightsRow * hierarchyVertices[0];
                coarseVertices[hierarchyVertices[0].rows() * 3]     = proj.x();
                coarseVertices[hierarchyVertices[0].rows() * 3 + 1] = proj.y();
                coarseVertices[hierarchyVertices[0].rows() * 3 + 2] = proj.z();
            } else if (useInvDistWeights) {
                auto p = surface.point(selectedPoint);
                coarseVertices[hierarchyVertices[0].rows() * 3]     = p.x();
                coarseVertices[hierarchyVertices[0].rows() * 3 + 1] = p.y();
                coarseVertices[hierarchyVertices[0].rows() * 3 + 2] = p.z();
            }

            int e = 0;
            for (int i = 0; i < hierarchyNeighbours[0].rows(); i++) {
                for (int j = 0; j < hierarchyNeighbours[0].cols(); j++) {
                    int to = hierarchyNeighbours[0](i, j);

                    if (to < 0) break;
                    if (i >= to) continue;

                    coarseEdges[e * 2]     = i;
                    coarseEdges[e * 2 + 1] = to;

                    if (useEdgeCoords && ((i == prolongationVertices[0] && to == prolongationVertices[1]) ||
                                          (i == prolongationVertices[1] && to == prolongationVertices[0]))) {
                        edgeCoordEdgeIndex = e;
                    }

                    e++;
                }
            }

            if (useBarycentricCoords || useInvDistWeights) {
                for (int i = 0; i < 3; i++) {
                    coarseEdges[(nCoarseEdges + i) * 2]     = hierarchyVertices[0].rows(); // selected point projection
                    coarseEdges[(nCoarseEdges + i) * 2 + 1] = prolongationVertices.at(i);
                }
            }

            int candidateTriangleIndex = 0;
            for (auto t : candidateTri) {
                auto i = candidateTriangleIndex++;
                candidateEdges[i * 6]     = t[0];
                candidateEdges[i * 6 + 1] = t[1];
                candidateEdges[i * 6 + 2] = t[1];
                candidateEdges[i * 6 + 3] = t[2];
                candidateEdges[i * 6 + 4] = t[2];
                candidateEdges[i * 6 + 5] = t[0];
            }

            // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
            glBindVertexArray(coarseVertexArray);

            glBindBuffer(GL_ARRAY_BUFFER, coarseVertexBuffer);
            glBufferData(GL_ARRAY_BUFFER, 3 * (hierarchyVertices[0].rows() + 1) * sizeof(GLfloat), coarseVertices, GL_STATIC_DRAW);

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
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * (nCoarseEdges + 3) * sizeof(GLuint), coarseEdges, GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseCandidateEdgeBuffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * candidateTri.size() * sizeof(GLuint), candidateEdges, GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, finalTriangleBuffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * sizeof(GLuint), &prolongationVertices[0], GL_STATIC_DRAW);
        }

        ImGui::End();

        ImGui::Begin("Prolongation", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration);
        ImGui::SetWindowPos(ImVec2(screenWidth - 300 - 10, 10));
        ImGui::SetWindowSize(ImVec2(300, 120));

        if (useBarycentricCoords) {
            ImGui::Text("Prolongation:");
            ImGui::Text("Barycentric coordinates in triangle");
            ImGui::Dummy(ImVec2(0.0f, 8.0f));
            ImGui::Text("v1: %f", prolongationWeights[0]);
            ImGui::Text("v2: %f", prolongationWeights[1]);
            ImGui::Text("v3: %f", prolongationWeights[2]);
        } else if (useEdgeCoords) {
            ImGui::Text("Prolongation:");
            ImGui::Text("Barycentric coordinates on edge");
            ImGui::Dummy(ImVec2(0.0f, 8.0f));
            ImGui::Text("v1: %f", prolongationWeights[0]);
            ImGui::Text("v2: %f", prolongationWeights[1]);
        } else {
            ImGui::Text("Prolongation:");
            ImGui::Text("Inverse distance weights");
            ImGui::Dummy(ImVec2(0.0f, 8.0f));
            ImGui::Text("v1: %f", prolongationWeights[0]);
            ImGui::Text("v2: %f", prolongationWeights[1]);
            ImGui::Text("v3: %f", prolongationWeights[2]);
        }

        ImGui::End();

        // input
        // -----
        processInput(window);

        int vpWidth = 0.333 * frameBufferWidth;
        int vpHeight = 0.9 * frameBufferHeight;

        // pass projection matrix to shader (note that in this case it could change every frame)
        projection = glm::perspective(glm::radians(camera.Zoom), (float)vpWidth / (float)vpHeight, 0.1f, 100.0f);
        ourShader.setMat4("projection", projection);

        // camera/view transformation
        view = camera.GetViewMatrix();
        ourShader.setMat4("view", view);

        // calculate the model matrix and pass it to shader before drawing
        model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        model = glm::rotate(model, rotationAngles.x + rotationAnglesDrag.x, glm::vec3(1, 0, 0));
        model = glm::rotate(model, rotationAngles.y + rotationAnglesDrag.y, glm::vec3(0, 1, 0));
        ourShader.setMat4("model", model);

        // set point size
        glPointSize(5 * 45 / camera.Zoom);

        // render
        // ------
        glClearColor(0.4f, 0.4f, 0.4f, 1.0f);
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
        glDrawElements(GL_LINES, 2 * nCoarseEdges, GL_UNSIGNED_INT, 0);

        // draw candidate edges
        ourShader.setBool("useUniformColor", true);
        ourShader.setVec3("uColor", glm::vec3(1, 1, 0));
        glDisableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseCandidateEdgeBuffer);
        glDrawElements(GL_LINES, 6 * candidateTri.size(), GL_UNSIGNED_INT, 0);

        if (useBarycentricCoords) {
            ourShader.setVec3("uColor", glm::vec3(0, 0.8f, 1));
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, finalTriangleBuffer);
            glDrawElements(GL_LINE_LOOP, 3, GL_UNSIGNED_INT, 0);
        } else if (useEdgeCoords) {
            ourShader.setVec3("uColor", glm::vec3(1, 0, 0));
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseEdgeBuffer);
            glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, (void*) (edgeCoordEdgeIndex * 2 * sizeof(GLuint)));
        }

        if (useBarycentricCoords || useInvDistWeights) {
            ourShader.setVec3("uColor", glm::vec3(1, 0, 0));
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseEdgeBuffer);
            glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, (void*) (nCoarseEdges * 2 * sizeof(GLuint)));
        }

        if (useBarycentricCoords || useEdgeCoords) {
            // draw projected point
            ourShader.setVec3("uColor", glm::vec3(0.6f, 0, 0.8f));
            glBindVertexArray(coarseVertexArray);
            glDrawArrays(GL_POINTS, hierarchyVertices[0].rows(), 1);

            // draw coarse point
            ourShader.setVec3("uColor", glm::vec3(1, 1, 0));
            glBindVertexArray(coarseVertexArray);
            glDrawArrays(GL_POINTS, coarsePoint, 1);
        }

        // draw selected point
        ourShader.setVec3("uColor", glm::vec3(1, 0, 1));
        glBindVertexArray(vertexArray);
        glDrawArrays(GL_POINTS, selectedPoint.idx(), 1);

        // draw vertices
        ourShader.setBool("useUniformColor", false);
        glEnableVertexAttribArray(1);
        glBindVertexArray(coarseVertexArray);
//        glDrawArrays(GL_POINTS, 0, coarserLevel.num_vertices());

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
    glDeleteBuffers(1, &coarseVertexArray);
    glDeleteBuffers(1, &coarseVertexBuffer);
    glDeleteBuffers(1, &coarseColorBuffer);
    glDeleteBuffers(1, &coarseEdgeBuffer);
    glDeleteBuffers(1, &coarseCandidateEdgeBuffer);
    glDeleteBuffers(1, &finalTriangleBuffer);

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

            if (rotationAnglesDrag == glm::vec2(0)) {
                int vpWidth = 0.333 * screenWidth;
                int vpHeight = 0.9 * screenHeight;

                if (currentMousePosition.x < vpWidth && currentMousePosition.y > 0.1 * screenHeight) {
                    float x = (2.0f * ((int)currentMousePosition.x % vpWidth)) / vpWidth - 1.0f;
                    float y = 1.0f - (2.0f * (currentMousePosition.y - 0.1f * screenHeight)) / vpHeight;
                    float z = 1.0f;
                    glm::vec3 ray_nds(x, y, z);

                    glm::vec4 ray_clip(ray_nds.x, ray_nds.y, -1.0, 1.0);

                    glm::vec4 ray_eye = inverse(projection) * ray_clip;

                    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0, 0.0);

                    glm::vec4 inv_ray_wor = inverse(model) * inverse(view) * ray_eye;
                    glm::vec3 ray_wor = glm::vec3(inv_ray_wor.x, inv_ray_wor.y, inv_ray_wor.z);
                    ray_wor = normalize(ray_wor);

                    rayCamPos = inverse(model) * glm::vec4(camera.Position.x, camera.Position.y, camera.Position.z, 1.0f);

                    float tMin = INFINITY;
                    SurfaceMesh::Vertex_index vMin;
                    bool intersection = false;
                    int intersections = 0;

                    float radius2 = 0.0001f;

                    for (auto v : surface.vertices()) {
                        float t0, t1; // solutions for t if the ray intersects

                        // geometric solution
                        auto vPos = toVec3(surface.point(v));
                        glm::vec3 L = vPos - rayCamPos;
                        float tca = glm::dot(L, ray_wor);
                        // if (tca < 0) return false;
                        float d2 = glm::dot(L, L) - tca * tca;
                        if (d2 > radius2) continue;
                        float thc = sqrt(radius2 - d2);
                        t0 = tca - thc;
                        t1 = tca + thc;

                        if (t0 > t1) std::swap(t0, t1);

                        if (t0 < 0) {
                            t0 = t1; // if t0 is negative, let's use t1 instead
                            if (t0 < 0) continue; // both t0 and t1 are negative
                        }

                        float t = t0;

                        if (t < tMin) {
                            tMin = t;
                            vMin = v;
                            intersection = true;
                            intersections++;
                        }
                    }

                    if (intersection) {
                        selectedPoint = vMin;
                    }
                }
            }

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

    currentMousePosition = glm::vec2(xpos, ypos);
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
