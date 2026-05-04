#include "AeroBridgeWorldSubsystem.h"

#include "Dom/JsonObject.h"
#include "HAL/IConsoleManager.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

namespace
{
bool EnsureGameWorld(UWorld* World, FOutputDevice& Ar)
{
	if (World == nullptr)
	{
		Ar.Log(TEXT("No world context available."));
		return false;
	}
	if (!World->IsGameWorld())
	{
		Ar.Log(TEXT("Aero bridge commands require PIE or game world."));
		return false;
	}
	return true;
}

UAeroBridgeWorldSubsystem* ResolveBridge(UWorld* World, FOutputDevice& Ar)
{
	if (!EnsureGameWorld(World, Ar))
	{
		return nullptr;
	}

	UAeroBridgeWorldSubsystem* Bridge = World->GetSubsystem<UAeroBridgeWorldSubsystem>();
	if (Bridge == nullptr)
	{
		Ar.Log(TEXT("AeroBridgeWorldSubsystem is unavailable."));
	}
	return Bridge;
}

FString JoinArgs(const TArray<FString>& Args)
{
	FString Joined;
	for (int32 Index = 0; Index < Args.Num(); ++Index)
	{
		if (Index > 0)
		{
			Joined.AppendChar(TEXT(' '));
		}
		Joined += Args[Index];
	}
	return Joined;
}

FString JoinArgsFrom(const TArray<FString>& Args, int32 StartIndex)
{
	FString Joined;
	for (int32 Index = StartIndex; Index < Args.Num(); ++Index)
	{
		if (Index > StartIndex)
		{
			Joined.AppendChar(TEXT(' '));
		}
		Joined += Args[Index];
	}
	return Joined;
}

FString MakeEnvelope(const TFunctionRef<void(TSharedPtr<FJsonObject>&)>& FillPayload)
{
	TSharedPtr<FJsonObject> RootObject = MakeShared<FJsonObject>();
	RootObject->SetStringField(TEXT("api_version"), TEXT("1.0"));
	RootObject->SetStringField(TEXT("request_id"), TEXT("console"));

	TSharedPtr<FJsonObject> PayloadObject = MakeShared<FJsonObject>();
	FillPayload(PayloadObject);
	RootObject->SetObjectField(TEXT("payload"), PayloadObject);

	FString Output;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
	FJsonSerializer::Serialize(RootObject.ToSharedRef(), Writer);
	return Output;
}

void LogResponse(FOutputDevice& Ar, const FString& Response)
{
	Ar.Log(*Response);
}

static FAutoConsoleCommand GAeroDescribeCapabilitiesCmd(
	TEXT("aero.describe_capabilities"),
	TEXT("aero.describe_capabilities [RequestJson]"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}

			const FString RequestJson = Args.Num() > 0 ? JoinArgs(Args) : TEXT("{}");
			LogResponse(Ar, Bridge->HandleDescribeCapabilities(RequestJson));
		}));

static FAutoConsoleCommand GAeroLoadContextCmd(
	TEXT("aero.load_context"),
	TEXT("aero.load_context <MapId>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.load_context <MapId>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}

			const FString RequestJson = MakeEnvelope(
				[&Args](TSharedPtr<FJsonObject>& PayloadObject)
				{
					PayloadObject->SetStringField(TEXT("map_id"), Args[0]);
				});
			LogResponse(Ar, Bridge->HandleLoadContext(RequestJson));
		}));

static FAutoConsoleCommand GAeroReloadConfigCmd(
	TEXT("aero.reload_config"),
	TEXT("aero.reload_config <Kind> [Path]"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.reload_config <Kind> [Path]"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}

			const FString RequestJson = MakeEnvelope(
				[&Args](TSharedPtr<FJsonObject>& PayloadObject)
				{
					PayloadObject->SetStringField(TEXT("kind"), Args[0]);
					if (Args.Num() > 1)
					{
						PayloadObject->SetStringField(TEXT("path"), JoinArgsFrom(Args, 1));
					}
				});
			LogResponse(Ar, Bridge->HandleReloadConfig(RequestJson));
		}));

static FAutoConsoleCommand GAeroApplyFrameJsonCmd(
	TEXT("aero.apply_frame_json"),
	TEXT("aero.apply_frame_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.apply_frame_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleApplyFrame(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroPollFeedbackJsonCmd(
	TEXT("aero.poll_feedback_json"),
	TEXT("aero.poll_feedback_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			const FString RequestJson = Args.Num() > 0 ? JoinArgs(Args) : TEXT("{}");

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandlePollFeedback(RequestJson));
		}));

static FAutoConsoleCommand GAeroSpawnAssetJsonCmd(
	TEXT("aero.spawn_asset_json"),
	TEXT("aero.spawn_asset_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.spawn_asset_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleSpawnAsset(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroMoveAssetJsonCmd(
	TEXT("aero.move_asset_json"),
	TEXT("aero.move_asset_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.move_asset_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleMoveAsset(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroRemoveAssetJsonCmd(
	TEXT("aero.remove_asset_json"),
	TEXT("aero.remove_asset_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.remove_asset_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleRemoveAsset(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroCaptureWorldCameraJsonCmd(
	TEXT("aero.capture_world_camera_json"),
	TEXT("aero.capture_world_camera_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.capture_world_camera_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleCaptureWorldCamera(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroReserveOccupancyJsonCmd(
	TEXT("aero.reserve_occupancy_json"),
	TEXT("aero.reserve_occupancy_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.reserve_occupancy_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleReserveOccupancy(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroReleaseOccupancyJsonCmd(
	TEXT("aero.release_occupancy_json"),
	TEXT("aero.release_occupancy_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.release_occupancy_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleReleaseOccupancy(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroQueryNearestJsonCmd(
	TEXT("aero.query_nearest_json"),
	TEXT("aero.query_nearest_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.query_nearest_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleQueryNearest(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroQueryPedPathJsonCmd(
	TEXT("aero.query_ped_path_json"),
	TEXT("aero.query_ped_path_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.query_ped_path_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleQueryPedPath(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroProjectGroundJsonCmd(
	TEXT("aero.project_ground_json"),
	TEXT("aero.project_ground_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.project_ground_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleProjectGround(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroQueryPedAnchorJsonCmd(
	TEXT("aero.query_ped_anchor_json"),
	TEXT("aero.query_ped_anchor_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.query_ped_anchor_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleQueryPedAnchor(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroApplyWeatherJsonCmd(
	TEXT("aero.apply_weather_json"),
	TEXT("aero.apply_weather_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.apply_weather_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleApplyWeather(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroCreateRuntimeMultirotorJsonCmd(
	TEXT("aero.create_runtime_multirotor_json"),
	TEXT("aero.create_runtime_multirotor_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.create_runtime_multirotor_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleCreateRuntimeMultirotor(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroMoveRuntimeMultirotorJsonCmd(
	TEXT("aero.move_runtime_multirotor_json"),
	TEXT("aero.move_runtime_multirotor_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.move_runtime_multirotor_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleMoveRuntimeMultirotor(JoinArgs(Args)));
		}));

static FAutoConsoleCommand GAeroRemoveRuntimeVehicleJsonCmd(
	TEXT("aero.remove_runtime_vehicle_json"),
	TEXT("aero.remove_runtime_vehicle_json <RequestJson>"),
	FConsoleCommandWithWorldArgsAndOutputDeviceDelegate::CreateStatic(
		[](const TArray<FString>& Args, UWorld* World, FOutputDevice& Ar)
		{
			if (Args.Num() < 1)
			{
				Ar.Log(TEXT("usage: aero.remove_runtime_vehicle_json <RequestJson>"));
				return;
			}

			UAeroBridgeWorldSubsystem* Bridge = ResolveBridge(World, Ar);
			if (Bridge == nullptr)
			{
				return;
			}
			LogResponse(Ar, Bridge->HandleRemoveRuntimeVehicle(JoinArgs(Args)));
		}));
}
