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

Eigen::MatrixXd originalPositions;
Eigen::MatrixXi originalNeighbours;

std::vector<int> prolongationVertices;
std::vector<double> prolongationWeights;


// vertex buffer: [n fine vertices, m coarse vertices, 1 projected vertex] x 3d
// sampling buffer [n fine points] x 1 bool
// color buffer: [n fine points] x 3f
// edge buffer: [n fine edges, m coarse edges] x 2i
// prolongation edge buffer: [n candidate edges, 3 final triangle edges, 3 barycentric lines] x 2i

unsigned int vertexArray, vertexBuffer, samplingBuffer, colorBuffer, edgeBuffer, prolongationEdgeBuffer;


enum Prolongation {
    barycentricTriangle,
    barycentricEdge,
    fallback,
};


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

    originalPositions = positions;
    originalNeighbours = neighbours;
}

static int countEdges(Eigen::MatrixXi &neighbours) {
    return ((neighbours.array() >= 0).cast<int>().sum()) / 2;
}

static std::vector<std::vector<int>> getEdges(Eigen::MatrixXi &neighbours) {
    std::vector<std::vector<int>> res;

    for (int i = 0; i < neighbours.rows(); i++) {
        for (int j = 0; j < neighbours.cols(); j++) {
            int to = neighbours(i, j);

            if (to < 0) break;
            if (i >= to) continue;

            res.push_back({ i, to });
        }
    }

    return res;
}

static void bufferHierarchyData(Eigen::MatrixXd &fineVertexPositions, Eigen::MatrixXd &coarseVertexPositions,
                         std::vector<int> &sampling, Eigen::VectorXi &neighbourhoods, std::vector<Eigen::Vector3f> &neighbourhoodColors,
                         Eigen::MatrixXi &fineNeighbours, Eigen::MatrixXi &coarseNeighbours) {
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, (void*)0);
    int vertexBufferSize = (fineVertexPositions.rows() + coarseVertexPositions.rows() + 1) * 3 * sizeof(GLdouble);
    glBufferData(GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_STATIC_DRAW);

    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> rowMajor = fineVertexPositions;
    int fineVertexDataSize = fineVertexPositions.rows() * 3 * sizeof(GLdouble);
    glBufferSubData(GL_ARRAY_BUFFER, 0, fineVertexDataSize, rowMajor.data());

    rowMajor = coarseVertexPositions;
    int coarseVertexDataSize = coarseVertexPositions.rows() * 3 * sizeof(GLdouble);
    glBufferSubData(GL_ARRAY_BUFFER, fineVertexDataSize, coarseVertexDataSize, rowMajor.data());

    glBindBuffer(GL_ARRAY_BUFFER, samplingBuffer);
    glVertexAttribIPointer(1, 1, GL_BYTE, 0, (void*)0);
    auto* samplingData = new GLboolean[fineVertexPositions.rows()]();
    for (int v : sampling) {
        samplingData[v] = 1;
    }
    glBufferData(GL_ARRAY_BUFFER, fineVertexPositions.rows() * sizeof(GLboolean), samplingData, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    std::vector<Eigen::Vector3f> colorData(fineVertexPositions.rows());
    for (int v = 0; v < fineVertexPositions.rows(); v++) {
        colorData[v] = neighbourhoodColors[neighbourhoods[v]];
    }
    glBufferData(GL_ARRAY_BUFFER, fineVertexPositions.rows() * sizeof(Eigen::Vector3f), &colorData[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, edgeBuffer);
    auto edges = getEdges(fineNeighbours);
    auto coarseEdges = getEdges(coarseNeighbours);
    for (auto& e : coarseEdges) {
        e[0] += fineVertexPositions.rows();
        e[1] += fineVertexPositions.rows();
    }
    edges.insert(edges.end(), coarseEdges.begin(), coarseEdges.end());
    std::vector<int> flattenedEdges;
    for (auto e : edges) {
        flattenedEdges.push_back(e[0]);
        flattenedEdges.push_back(e[1]);
    }
    glBufferData(GL_ARRAY_BUFFER, edges.size() * 2 * sizeof(GLuint), &flattenedEdges[0], GL_STATIC_DRAW);
}

static void bufferProlongationData(Prolongation prolongation, Eigen::MatrixXd &fineVertexPositions, Eigen::MatrixXd &coarseVertexPositions,
                                   Eigen::Vector3d &projectedVertexPosition, int finePoint, std::set<std::vector<int>> &candidateEdges, std::vector<int> &prolongationVertices) {
    glBindVertexArray(vertexArray);

    if (prolongation != fallback) {
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        int projectedVertexOffset = (fineVertexPositions.rows() + coarseVertexPositions.rows()) * 3 * sizeof(GLdouble);
        glBufferSubData(GL_ARRAY_BUFFER, projectedVertexOffset, 3 * sizeof(GLdouble), projectedVertexPosition.data());
    }

    glBindBuffer(GL_ARRAY_BUFFER, prolongationEdgeBuffer);

    std::vector<std::vector<int>> prolongationEdges(candidateEdges.begin(), candidateEdges.end());

    int nFinePoints = fineVertexPositions.rows();
    for (auto& e : prolongationEdges) {
        e[0] += nFinePoints;
        e[1] += nFinePoints;
    }

    int projectedVertex = fineVertexPositions.rows() + coarseVertexPositions.rows();

    switch (prolongation) {
        case barycentricTriangle:
            for (int i = 0; i < 3; i++) {
                prolongationEdges.push_back({ prolongationVertices[i] + nFinePoints, prolongationVertices[(i + 1) % 3] + nFinePoints }); // final triangle
            }
            for (int i = 0; i < 3; i++) {
                prolongationEdges.push_back({ projectedVertex, prolongationVertices[i] + nFinePoints }); // barycentric lines
            }
            break;
        case barycentricEdge:
            prolongationEdges.push_back({ prolongationVertices[0] + nFinePoints, prolongationVertices[1] + nFinePoints });
            break;
        case fallback:
            for (int i = 0; i < 3; i++) {
                prolongationEdges.push_back({ finePoint, prolongationVertices[i] + nFinePoints }); // lines to closest points
            }
            break;
    }

    std::vector<int> flattenedEdges;
    for (auto e : prolongationEdges) {
        flattenedEdges.push_back(e[0]);
        flattenedEdges.push_back(e[1]);
    }

    glBufferData(GL_ARRAY_BUFFER, prolongationEdges.size() * 2 * sizeof(GLuint), &flattenedEdges[0], GL_STATIC_DRAW);
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
    Shader defaultShader("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl");
    Shader samplingShader("shaders/vertex_shader_sampling.glsl", "shaders/fragment_shader.glsl");

    samplingShader.setVec3("samplingColor", 1, 1, 0);
    samplingShader.setVec3("selectionColor", 1, 0, 1);

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

    glGenVertexArrays(1, &vertexArray);
    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &samplingBuffer);
    glGenBuffers(1, &colorBuffer);
    glGenBuffers(1, &edgeBuffer);
    glGenBuffers(1, &prolongationEdgeBuffer);

    std::set<std::vector<int>> candidateEdges;
    Prolongation prolongation;

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

            std::vector<Eigen::Vector3f> neighbourhoodColors(hierarchyVertices[0].rows());
            for (int neighbourhood = 0; neighbourhood < hierarchyVertices[0].rows(); neighbourhood++) {
                neighbourhoodColors[neighbourhood] = Eigen::Vector3f(((neighbourhood * 3) % 255) / 255.0f,
                                                                     ((neighbourhood * 5) % 255) / 255.0f,
                                                                     ((neighbourhood * 7) % 255) / 255.0f);
            }

            Eigen::VectorXi neighbourhoods(hierarchyProlongation[0].rows());
            for (int v = 0; v < hierarchyProlongation[0].rows(); v++) {
                Eigen::RowVectorXd weightsRow = hierarchyProlongation[0].row(v);
                Eigen::RowVectorXd::Index maxIndex;
                weightsRow.maxCoeff(&maxIndex);
                neighbourhoods[v] = maxIndex;
            }

            coarsePoint = neighbourhoods[selectedPoint];

            Eigen::RowVectorXd weightsRow = hierarchyProlongation[0].row(selectedPoint);

            int nElements = (weightsRow.array() > 0.0).cast<int>().sum();

            if (hierarchyProlongationFallback[0][selectedPoint]) {
                prolongation = fallback;
            } else if (nElements == 3) {
                prolongation = barycentricTriangle;
            } else {
                prolongation = barycentricEdge;
            }

            prolongationVertices.clear();
            prolongationWeights.clear();

            for (int j = 0; j < hierarchyProlongation[0].cols(); j++) {
                if (weightsRow[j] > 0) {
                    prolongationVertices.push_back(j);
                    prolongationWeights.push_back(weightsRow[j]);
                }
            }

            if (nElements == 1) {
                prolongationVertices[1] = prolongationVertices[0];
                prolongationWeights[1] = 0;
            }

            auto contains = [](auto v, auto x) { return std::find(v.begin(), v.end(), x) != v.end(); };

            candidateEdges.clear();
            for (auto t : hierarchyTriangles[0]) {
                if (!contains(t, coarsePoint)) continue;

                candidateEdges.insert({ std::min(t[0], t[1]), std::max(t[0], t[1]) });
                candidateEdges.insert({ std::min(t[1], t[2]), std::max(t[1], t[2]) });
                candidateEdges.insert({ std::min(t[2], t[0]), std::max(t[2], t[0]) });
            }

            Eigen::Vector3d projectedVertexPosition = weightsRow * hierarchyVertices[0];

            bufferHierarchyData(originalPositions, hierarchyVertices[0], hierarchySampling[0], neighbourhoods, neighbourhoodColors, originalNeighbours, hierarchyNeighbours[0]);
            bufferProlongationData(prolongation, originalPositions, hierarchyVertices[0], projectedVertexPosition, selectedPoint, candidateEdges, prolongationVertices);

            samplingShader.setInt("selectedVertex", selectedPoint);
        }

        ImGui::End();

        ImGui::Begin("Prolongation", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration);
        ImGui::SetWindowPos(ImVec2(screenWidth - 300 - 10, 10));
        ImGui::SetWindowSize(ImVec2(300, 120));

        const char* prolongationDesc;
        switch (prolongation) {
            case barycentricTriangle:
                prolongationDesc = "Barycentric coordinates in triangle";
                break;
            case barycentricEdge:
                prolongationDesc = "Barycentric coordinates on edge";
                break;
            case fallback:
                prolongationDesc = "Inverse distance weights";
                break;
        }

        ImGui::Text("Prolongation:");
        ImGui::Text("%s", prolongationDesc);
        ImGui::Dummy(ImVec2(0.0f, 8.0f));
        ImGui::Text("v1: %f", prolongationWeights[0]);
        ImGui::Text("v2: %f", prolongationWeights[1]);

        if (prolongation != barycentricEdge) {
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
        defaultShader.setMat4("projection", projection);
        samplingShader.setMat4("projection", projection);

        // camera/view transformation
        view = camera.GetViewMatrix();
        defaultShader.setMat4("view", view);
        samplingShader.setMat4("view", view);

        // calculate the model matrix and pass it to shader before drawing
        model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        model = glm::rotate(model, rotationAngles.x + rotationAnglesDrag.x, glm::vec3(1, 0, 0));
        model = glm::rotate(model, rotationAngles.y + rotationAnglesDrag.y, glm::vec3(0, 1, 0));
        defaultShader.setMat4("model", model);
        samplingShader.setMat4("model", model);

        // set point size
        glPointSize(5 * 45 / camera.Zoom);

        // render
        // ------
        glClearColor(0.4f, 0.4f, 0.4f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        int nFineEdges = countEdges(originalNeighbours);
        int nCoarseEdges = countEdges(hierarchyNeighbours[0]);

        glBindVertexArray(vertexArray);
        glEnableVertexAttribArray(0);

        // sampling (left pane)
        glViewport(0, 0, vpWidth, vpHeight);

        // draw edges
        defaultShader.use();
        defaultShader.setBool("useUniformColor", false);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
        glDrawElements(GL_LINES, 2 * nFineEdges, GL_UNSIGNED_INT, 0);

        // draw vertices
        samplingShader.use();
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glDrawArrays(GL_POINTS, 0, originalPositions.rows());
        glDisableVertexAttribArray(1);

        // neighbourhoods (middle pane)
        glViewport(vpWidth, 0, vpWidth, vpHeight);

        // draw edges
        defaultShader.use();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
        glDrawElements(GL_LINES, 2 * nFineEdges, GL_UNSIGNED_INT, 0);

        // draw vertices
        glEnableVertexAttribArray(2);
        glDrawArrays(GL_POINTS, 0, originalPositions.rows());
        glDisableVertexAttribArray(2);

        // prolongation (right pane)
        glViewport(2 * vpWidth, 0, vpWidth, vpHeight);

        // draw edges
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
        glDrawElements(GL_LINES, 2 * nCoarseEdges, GL_UNSIGNED_INT, (void*) (nFineEdges * 2 * sizeof(GLuint)));

        // draw candidate edges
        defaultShader.setBool("useUniformColor", true);
        defaultShader.setVec3("uColor", glm::vec3(1, 1, 0));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prolongationEdgeBuffer);
        glDrawElements(GL_LINES, 2 * candidateEdges.size(), GL_UNSIGNED_INT, 0);

        switch (prolongation) {
            case barycentricTriangle:
                // draw final triangle
                defaultShader.setVec3("uColor", glm::vec3(0, 0.8f, 1));
                glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, (void*) (candidateEdges.size() * 2 * sizeof(GLuint)));

                // draw barycentric lines
                defaultShader.setVec3("uColor", glm::vec3(1, 0, 0));
                glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, (void*) ((candidateEdges.size() + 3) * 2 * sizeof(GLuint)));
                break;
            case barycentricEdge:
                // draw barycentric edge
                defaultShader.setVec3("uColor", glm::vec3(1, 0, 0));
                glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, (void*) (candidateEdges.size() * 2 * sizeof(GLuint)));
                break;
            case fallback:
                // draw lines to closest points
                defaultShader.setVec3("uColor", glm::vec3(1, 0, 0));
                glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, (void*) (candidateEdges.size() * 2 * sizeof(GLuint)));
                break;
        }

        if (prolongation != fallback) {
            // draw coarse point
            defaultShader.setVec3("uColor", glm::vec3(1, 1, 0));
            glDrawArrays(GL_POINTS, originalPositions.rows() + coarsePoint, 1);

            // draw projected point
            defaultShader.setVec3("uColor", glm::vec3(0.6f, 0, 0.8f));
            glDrawArrays(GL_POINTS, originalPositions.rows() + hierarchyVertices[0].rows(), 1);
        }

        // draw selected point
        defaultShader.setVec3("uColor", glm::vec3(1, 0, 1));
        glDrawArrays(GL_POINTS, selectedPoint.idx(), 1);

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
    glDeleteBuffers(1, &samplingBuffer);
    glDeleteBuffers(1, &colorBuffer);
    glDeleteBuffers(1, &edgeBuffer);
    glDeleteBuffers(1, &prolongationEdgeBuffer);

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
