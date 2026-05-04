#include "AeroEditorToolsSubsystem.h"

#include "Containers/Ticker.h"
#include "Editor.h"
#include "HAL/IConsoleManager.h"
#include "HAL/PlatformMisc.h"
#include "Misc/CommandLine.h"
#include "Modules/ModuleManager.h"
#include "Misc/Parse.h"

namespace
{
UAeroEditorToolsSubsystem* ResolveEditorTools(FOutputDevice& Ar)
{
	if (GEditor == nullptr)
	{
		Ar.Log(TEXT("GEditor is unavailable."));
		return nullptr;
	}

	UAeroEditorToolsSubsystem* Subsystem = GEditor->GetEditorSubsystem<UAeroEditorToolsSubsystem>();
	if (Subsystem == nullptr)
	{
		Ar.Log(TEXT("AeroEditorToolsSubsystem is unavailable."));
	}

	return Subsystem;
}

void RunBootstrap(FOutputDevice& Ar)
{
	UAeroEditorToolsSubsystem* Subsystem = ResolveEditorTools(Ar);
	if (Subsystem == nullptr)
	{
		return;
	}

	FString Error;
	if (!Subsystem->BootstrapAeroWorldContentAssets(Error))
	{
		Ar.Logf(TEXT("BootstrapAeroWorldContentAssets failed: %s"), Error.IsEmpty() ? TEXT("unknown error") : *Error);
		return;
	}

	Ar.Log(TEXT("BootstrapAeroWorldContentAssets succeeded."));
}

void RunValidation(FOutputDevice& Ar)
{
	UAeroEditorToolsSubsystem* Subsystem = ResolveEditorTools(Ar);
	if (Subsystem == nullptr)
	{
		return;
	}

	FString Error;
	if (!Subsystem->ValidateAeroWorldContentAssets(Error))
	{
		Ar.Logf(TEXT("ValidateAeroWorldContentAssets failed: %s"), Error.IsEmpty() ? TEXT("unknown error") : *Error);
		return;
	}

	Ar.Log(TEXT("ValidateAeroWorldContentAssets succeeded."));
}

bool HasHeadlessBootstrapFlag()
{
	return FParse::Param(FCommandLine::Get(), TEXT("AeroBootstrapWorldContent"));
}

bool HasHeadlessValidateFlag()
{
	return FParse::Param(FCommandLine::Get(), TEXT("AeroValidateWorldContent"));
}

static FAutoConsoleCommandWithOutputDevice GAeroBootstrapWorldContentCmd(
	TEXT("aero.bootstrap_world_content"),
	TEXT("Generate or update authoritative AeroWorldContent blueprint and data assets."),
	FConsoleCommandWithOutputDeviceDelegate::CreateStatic(RunBootstrap));

static FAutoConsoleCommandWithOutputDevice GAeroValidateWorldContentCmd(
	TEXT("aero.validate_world_content"),
	TEXT("Validate authoritative AeroWorldContent assets and asset catalog mappings."),
	FConsoleCommandWithOutputDeviceDelegate::CreateStatic(RunValidation));
} // namespace

class FAeroEditorToolsModule : public IModuleInterface
{
public:
	virtual void StartupModule() override
	{
		if (!HasHeadlessBootstrapFlag() && !HasHeadlessValidateFlag())
		{
			return;
		}

		FTSTicker::GetCoreTicker().AddTicker(
			FTickerDelegate::CreateLambda(
				[](float)
				{
					if (GEditor == nullptr)
					{
						return true;
					}

					UAeroEditorToolsSubsystem* Subsystem = GEditor->GetEditorSubsystem<UAeroEditorToolsSubsystem>();
					if (Subsystem == nullptr)
					{
						UE_LOG(LogTemp, Error, TEXT("Headless AeroEditorTools run failed: subsystem unavailable."));
						FPlatformMisc::RequestExit(false);
						return false;
					}

					bool bSucceeded = true;
					FString Error;
					if (HasHeadlessBootstrapFlag())
					{
						bSucceeded &= Subsystem->BootstrapAeroWorldContentAssets(Error);
						if (bSucceeded)
						{
							UE_LOG(LogTemp, Display, TEXT("Headless BootstrapAeroWorldContentAssets succeeded"));
						}
						else
						{
							UE_LOG(LogTemp, Error, TEXT("Headless BootstrapAeroWorldContentAssets failed: %s"), *Error);
						}
					}

					if (bSucceeded && HasHeadlessValidateFlag())
					{
						Error.Reset();
						bSucceeded &= Subsystem->ValidateAeroWorldContentAssets(Error);
						if (bSucceeded)
						{
							UE_LOG(LogTemp, Display, TEXT("Headless ValidateAeroWorldContentAssets succeeded"));
						}
						else
						{
							UE_LOG(LogTemp, Error, TEXT("Headless ValidateAeroWorldContentAssets failed: %s"), *Error);
						}
					}

					FPlatformMisc::RequestExit(false);
					return false;
				}),
			0.0f);
	}
};

IMPLEMENT_MODULE(FAeroEditorToolsModule, AeroEditorTools)
